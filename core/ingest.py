import os
import json
import base64
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

# Setup
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

DRAWINGS_FOLDER = "./drawings"

EXTRACTION_PROMPT = """
You are an expert mechanical engineering drawing analyser.
Look at this engineering drawing carefully.

Identify the component_type by determining what the component actually is.
Use your mechanical engineering knowledge to classify it accurately.
Provide a short, precise label such as "spur gear", "stepped shaft", "mounting bracket", "bearing housing", "connecting rod", "impeller", "rocker arm", "cam", "valve body", or any other appropriate term.

If the drawing shows an assembly, classify based on the PRIMARY component in the main detailed view.

Examples of classification logic:
- A cylindrical sleeve with a bore is a housing, not a bracket.
- A long rotating cylindrical component with varying diameters and keyways is a shaft, not a cylinder.

Extract the following and return ONLY a valid JSON object with no explanation and no markdown backticks.

{
  "component_type": "short precise label describing the component",
  "material": "material if mentioned, otherwise unknown",
  "drawing_number": "drawing number if visible, otherwise unknown",
  "scale": "scale if mentioned, otherwise unknown",
  "dimensions": ["list all dimensions and measurements you can see with units"],
  "tolerances": ["list all tolerances if visible"],
  "surface_finish": "surface finish specification if visible, otherwise unknown",
  "revision": "revision number or letter if visible, otherwise unknown",
  "notes": ["any notes or special instructions visible on the drawing"],
  "units": "mm or inches or unknown"
}
"""

def get_chroma_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="drawings")

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_media_type(filename):
    ext = filename.lower().split(".")[-1]
    mapping = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp"
    }
    return mapping.get(ext, "image/jpeg")

def extract_drawing_data(image_path, filename):
    """
    Sends image to Groq Vision and extracts structured metadata.
    Returns parsed JSON dictionary.
    """
    base64_image = encode_image(image_path)
    media_type = get_media_type(filename)

    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())

def build_text_representation(data):
    """
    Converts extracted JSON into a single text string for embedding.
    """
    return f"""
    Component: {data.get('component_type', 'unknown')}
    Material: {data.get('material', 'unknown')}
    Drawing Number: {data.get('drawing_number', 'unknown')}
    Scale: {data.get('scale', 'unknown')}
    Dimensions: {', '.join(data.get('dimensions', []))}
    Tolerances: {', '.join(data.get('tolerances', []))}
    Surface Finish: {data.get('surface_finish', 'unknown')}
    Units: {data.get('units', 'unknown')}
    Notes: {', '.join(data.get('notes', []))}
    """

def store_drawing(collection, image_path, filename, data):
    """
    Stores extracted drawing data and its embedding in ChromaDB.
    Deletes existing entry first if it already exists.
    """
    text = build_text_representation(data)
    embedding = embedder.encode(text).tolist()

    existing = collection.get(ids=[filename])
    if existing["ids"]:
        collection.delete(ids=[filename])

    collection.add(
        ids=[filename],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "filename": filename,
            "image_path": image_path,
            "component_type": data.get("component_type", "unknown"),
            "material": data.get("material", "unknown"),
            "drawing_number": data.get("drawing_number", "unknown"),
            "scale": data.get("scale", "unknown"),
            "units": data.get("units", "unknown"),
            "raw_json": json.dumps(data)
        }]
    )

def ingest_single(image_path, filename):
    """
    Full pipeline for one drawing.
    Returns (success, data or error message).
    """
    try:
        collection = get_chroma_collection()
        data = extract_drawing_data(image_path, filename)
        store_drawing(collection, image_path, filename, data)
        return True, data
    except Exception as e:
        return False, str(e)

def ingest_all():
    """
    Ingests all drawings in the drawings folder.
    Used for bulk ingestion from command line.
    """
    collection = get_chroma_collection()
    files = os.listdir(DRAWINGS_FOLDER)
    image_files = [
        f for f in files
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]

    print(f"Found {len(image_files)} drawings")

    for filename in image_files:
        image_path = os.path.join(DRAWINGS_FOLDER, filename)
        print(f"Processing: {filename}")
        success, result = ingest_single(image_path, filename)
        if success:
            print(f"Stored: {result.get('component_type')} | {result.get('material')}")
        else:
            print(f"Error: {result}")

    print(f"\nDone. {len(image_files)} drawings processed.")

if __name__ == "__main__":
    ingest_all()