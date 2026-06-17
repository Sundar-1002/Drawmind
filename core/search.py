import os
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
    Returns total drawing count and component type breakdown.
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
    Returns component type string or None if question is not component specific.
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

    # take only the first word to handle any extra output
    result = response.choices[0].message.content.strip().lower().split()[0]
    
    # clean punctuation
    result = result.strip(".,!?")
    
    return None if result == "none" else result

def build_context_from_all():
    """
    Fetches all drawings from ChromaDB and builds context string.
    Used for global questions about the entire database.
    """
    collection = get_chroma_collection()
    all_docs = collection.get(include=["documents", "metadatas"])
    context = ""
    for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
        raw = json.loads(meta.get("raw_json", "{}"))
        context += f"\nDrawing: {meta['filename']}\n"
        context += json.dumps(raw, indent=2)
        context += "\n"
    return context, len(all_docs["documents"])

def build_context_by_component(component_type):
    """
    Directly filters ChromaDB by component type metadata.
    Much more accurate than semantic search for component specific questions.
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

def build_context_from_search(question):
    """
    Searches for relevant drawings and builds context string.
    Used for non component specific questions.
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
    Automatically detects global vs specific questions.
    For component specific questions, directly filters by metadata.
    For other questions, uses semantic search.
    Returns answer string and referenced metadata list.
    """
    # initialize to empty by default
    referenced_metadatas = []
    context = ""
    count = 0

    if is_global_question(question):
        context, count = build_context_from_all()
        referenced_metadatas = []
    else:
        matched_component = extract_component_from_question(question)
        print(f"Matched component: {matched_component}")

        if matched_component and matched_component != "none":
            context, count, referenced_metadatas = build_context_by_component(matched_component)

            if count == 0:
                context = "No drawings found for this component type."
                referenced_metadatas = []
        else:
            context, count, referenced_metadatas = build_context_from_search(question)

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
                Be specific and mention drawing filenames when relevant."""
            },
            {
                "role": "user",
                "content": f"Based on these engineering drawings:\n{context}\n\nQuestion: {question}"
            }
        ],
        max_tokens=600
    )

    answer = response.choices[0].message.content
    return answer, referenced_metadatas 