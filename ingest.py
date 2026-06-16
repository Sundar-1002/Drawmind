import os
import json
import base64
import chromadb
from PIL import Image
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

# Setup Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Setup ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="drawings")

# Setup embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

DRAWINGS_FOLDER = "./drawings"

EXTRACTION_PROMPT = """
You are an engineering drawing analyser.
Look at this engineering drawing carefully and extract the following information.
Return ONLY a valid JSON object with no explanation and no markdown backticks.

{
  "component_type": "what type of component is this (gear, shaft, bracket, flange etc)",
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

def process_drawing(image_path, filename):
    print(f"Processing: {filename}")

    try:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        ext = filename.lower().split(".")[-1]
        media_type_map = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp"
        }
        media_type = media_type_map.get(ext, "image/jpeg")

        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": EXTRACTION_PROMPT
                        },
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

        data = json.loads(raw.strip())

        text = f"""
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

        embedding = embedder.encode(text).tolist()

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

        print(f"Successfully stored: {filename}")
        print(f"Component: {data.get('component_type')} | Material: {data.get('material')}")
        print("---")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

def ingest_all():
    files = os.listdir(DRAWINGS_FOLDER)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    print(f"Found {len(image_files)} drawings to process")
    print("Starting ingestion...\n")

    for filename in image_files:
        image_path = os.path.join(DRAWINGS_FOLDER, filename)
        process_drawing(image_path, filename)

    print(f"\nIngestion complete. {len(image_files)} drawings processed.")

if __name__ == "__main__":
    ingest_all()