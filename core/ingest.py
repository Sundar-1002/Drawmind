import os
import json
import base64
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import cloudinary
import cloudinary.uploader

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)


def upload_to_cloudinary(image_path, filename):
    """
    Uploads image to Cloudinary and returns public URL.
    """
    result = cloudinary.uploader.upload(
        image_path,
        public_id=f"drawmind/{filename}",
        overwrite=True
    )
    return result["secure_url"]

# Setup
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Pinecone setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX", "drawmind"))

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

def encode_image(image_path):
    """
    Encodes image file to base64 string for Groq Vision API.

    Args:
        image_path (str): Full path to image file.

    Returns:
        str: Base64 encoded string.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_media_type(filename):
    """
    Returns MIME type for a given image filename.

    Args:
        filename (str): Image filename with extension.

    Returns:
        str: MIME type string.
    """
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

    Args:
        image_path (str): Full path to drawing image.
        filename (str): Image filename for media type detection.

    Returns:
        dict: Extracted fields including component_type, material,
              dimensions, tolerances, surface_finish, notes, units.
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
    Converts extracted JSON into a flat text string for embedding.

    Args:
        data (dict): Extracted drawing data from extract_drawing_data().

    Returns:
        str: Formatted multi-line text combining all key fields.
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

def store_drawing(image_path, filename, data):
    """
    Stores extracted drawing data and embedding in Pinecone.
    Uses filename as unique vector ID so re-ingesting updates existing entry.

    Args:
        image_path (str): Full path to drawing image.
        filename (str): Unique identifier for this drawing in Pinecone.
        data (dict): Extracted drawing data from extract_drawing_data().
    """
    text = build_text_representation(data)
    embedding = embedder.encode(text).tolist()

    # pinecone metadata must be flat key value pairs
    # store raw_json as string for later retrieval

    image_url = upload_to_cloudinary(image_path, filename) 

    index.upsert(
        vectors=[{
            "id": filename,
            "values": embedding,
            "metadata": {
                "filename": filename,
                "image_url": image_url,
                "component_type": data.get("component_type", "unknown"),
                "material": data.get("material", "unknown"),
                "drawing_number": data.get("drawing_number", "unknown"),
                "scale": data.get("scale", "unknown"),
                "units": data.get("units", "unknown"),
                "raw_json": json.dumps(data)
            }
        }]
    )

def ingest_single(image_path, filename):
    """
    Full ingestion pipeline for one drawing.
    Extracts metadata with Vision AI and stores in Pinecone.

    Args:
        image_path (str): Full path to drawing image.
        filename (str): Image filename.

    Returns:
        tuple: (success, data or error message)
    """
    try:
        data = extract_drawing_data(image_path, filename)
        store_drawing(image_path, filename, data)
        return True, data
    except Exception as e:
        return False, str(e)

def ingest_all():
    """
    Bulk ingestion of all drawings in DRAWINGS_FOLDER.
    Processes every image and stores in Pinecone.
    Used for initial database population from command line.
    """
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