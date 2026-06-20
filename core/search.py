import os
import re
import json
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

# Setup
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_chroma_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="drawings")

def get_database_stats():
    """
    Returns total drawing count and component type breakdown from ChromaDB.

    Returns:
        tuple: (total, component_types) where total is int and
               component_types is dict mapping type names to counts.
    """
    collection = get_chroma_collection()
    total = collection.count()
    all_data = collection.get()

    component_types = {}
    for meta in all_data["metadatas"]:
        ct = meta.get("component_type", "unknown").lower()
        component_types[ct] = component_types.get(ct, 0) + 1

    return total, component_types

def is_global_question(question):
    """
    Detects if question is about the entire database
    rather than a specific drawing.
    """
    global_keywords = [
        "all", "available", "database", "total",
        "how many", "list", "types", "what components",
        "every", "entire"
    ]
    return any(kw in question.lower() for kw in global_keywords)

def extract_component_from_question(question):
    """
    Uses LLM to extract which component type the question is about.
    Returns component type string or None if not component specific.
    """
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """You are a mechanical component classifier.
                Look at the question and return ONLY the component type word.
                Examples:
                "give me gear drawings" -> gear
                "show me shaft components" -> shaft
                "find fastener drawings" -> fastener
                "which drawing has alloy steel" -> none
                "what is the diameter" -> none
                Return ONE word only. No sentences. No explanation."""
            },
            {
                "role": "user",
                "content": question
            }
        ],
        max_tokens=5
    )

    result = response.choices[0].message.content.strip().lower().split()[0]
    result = result.strip(".,!?")
    return None if result == "none" else result

def extract_filename_from_question(question):
    """
    Checks if the question references a specific drawing filename.
    Returns filename string or None.
    Example: "give me image of 8.jpg" -> "8.jpg"
    """
    match = re.search(r'\b\d+[\w]*\.\w+\b', question)
    if match:
        return match.group(0)
    return None

def build_context_from_all():
    """
    Fetches all drawings as compact summary for global questions.
    Avoids token limit issues with 30 plus drawings.
    """
    collection = get_chroma_collection()
    all_docs = collection.get(include=["metadatas"])

    context = ""
    for meta in all_docs["metadatas"]:
        context += f"Drawing: {meta['filename']} | Component: {meta['component_type']} | Material: {meta['material']} | Drawing No: {meta['drawing_number']} | Scale: {meta['scale']} | Units: {meta['units']}\n"

    return context, len(all_docs["metadatas"])

def build_context_by_component(component_type):
    """
    Directly filters ChromaDB by component type metadata.
    More accurate than semantic search for component specific questions.
    """
    collection = get_chroma_collection()
    all_docs = collection.get(include=["documents", "metadatas"])

    filtered_docs = []
    filtered_metas = []

    for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
        if component_type in meta.get("component_type", "").lower():
            filtered_docs.append(doc)
            filtered_metas.append(meta)

    context = ""
    for doc, meta in zip(filtered_docs, filtered_metas):
        raw = json.loads(meta.get("raw_json", "{}"))
        context += f"\nDrawing: {meta['filename']}\n"
        context += json.dumps(raw, indent=2)
        context += "\n"

    return context, len(filtered_docs), filtered_metas

def build_context_by_filename(filename):
    """
    Directly fetches a specific drawing by filename from ChromaDB.
    """
    collection = get_chroma_collection()
    result = collection.get(
        ids=[filename],
        include=["documents", "metadatas"]
    )

    if not result["ids"]:
        return "", 0, []

    meta = result["metadatas"][0]
    raw = json.loads(meta.get("raw_json", "{}"))
    context = f"\nDrawing: {meta['filename']}\n"
    context += json.dumps(raw, indent=2)

    return context, 1, [meta]

def build_context_from_search(question):
    """
    Semantic search for non component specific questions.
    """
    collection = get_chroma_collection()
    query_embedding = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    context = ""
    for doc, meta in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        raw = json.loads(meta.get("raw_json", "{}"))
        context += f"\nDrawing: {meta['filename']}\n"
        context += json.dumps(raw, indent=2)
        context += "\n"
    return context, len(results["documents"][0]), results["metadatas"][0]

def ask_question(question):
    """
    Answers a natural language question about the drawings.
    Routes question to correct retrieval strategy.
    LLM decides whether to include image references via IMAGES: line.
    Returns answer string and referenced metadata list.
    """
    # initialize all variables at top
    referenced_metadatas = []
    all_metadatas = []
    context = ""
    count = 0

    if is_global_question(question):
        context, count = build_context_from_all()
        all_metadatas = []

    else:
        # check for specific filename first
        specific_file = extract_filename_from_question(question)

        if specific_file:
            context, count, all_metadatas = build_context_by_filename(specific_file)
            if count == 0:
                context = f"No drawing found with filename {specific_file}."
                all_metadatas = []

        else:
            # check for component type
            matched_component = extract_component_from_question(question)

            if matched_component and matched_component != "none":
                context, count, all_metadatas = build_context_by_component(matched_component)
                if count == 0:
                    context = "No drawings found for this component type."
                    all_metadatas = []
            else:
                context, count, all_metadatas = build_context_from_search(question)

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": f"""You are an engineering drawing assistant.
                Answer questions based ONLY on the provided drawing data.
                There are exactly {count} drawings in the context.
                Do NOT invent, duplicate, or assume any information not present.
                If asked for more samples than available, clearly state how many are available.
                Never include raw JSON or code in your answer. Always answer in clean natural language.

                IMAGES RULE:
                At the end of your answer add a line starting with IMAGES: in these cases:
                1. The question uses words like: show, display, give, find, retrieve, see, view, image, images, drawing, drawings, what does, look like
                2. The question asks to find or retrieve specific drawings by filename
                3. The question asks to list or show components of a specific type

                For analytical questions like "which has largest dimension", "compare", "how many",
                do NOT add an IMAGES line. Just answer in text.

                When you do add IMAGES, list ONLY the filenames that directly answer the question.

                Example for "show me all images":
                IMAGES: 1..png, 2.png, 3.jpg, 4.webp, 5.png ...all filenames

                Example for "show me all gear drawings":
                IMAGES: 1..png, 2.png, 3.jpg, 4.webp, 5.png, 6.png, 7.png, 8.jpg

                Example for "give me gears greater than 80mm":
                The gear with dimensions greater than 80mm is Drawing 6.png with outside diameter 167.4mm.
                IMAGES: 6.png

                Example for "which drawing has the largest dimension":
                Drawing 12.png has the largest dimension of 232mm.
                (no IMAGES line)"""
            },
            {
                "role": "user",
                "content": f"Based on these engineering drawings:\n{context}\n\nQuestion: {question}"
            }
        ],
        max_tokens=500
    )

    raw_answer = response.choices[0].message.content

    # parse IMAGES line if LLM included one
    if "IMAGES:" in raw_answer:
        parts = raw_answer.split("IMAGES:")
        answer = parts[0].strip()
        images_line = parts[1].strip()
        image_filenames = [f.strip() for f in images_line.split(",") if f.strip()]
        referenced_metadatas = [
            m for m in all_metadatas
            if m.get("filename") in image_filenames
        ]
    else:
        answer = raw_answer
        referenced_metadatas = []

    return answer, referenced_metadatas