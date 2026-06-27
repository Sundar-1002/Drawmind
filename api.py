import os
import json 
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.ingest import ingest_single, DRAWINGS_FOLDER
from core.search import ask_question, get_database_stats, get_chroma_collection

# FastAPI application instance
app = FastAPI(
    title="DrawMind API",
    description = "Engineering Drawing Intelligence System API",
    version="1.0.0"
)

# CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
def health():
    """
    Health check endpoint to verify that the API is running.
    """
    return {
        "status": "ok",
        "version": "1.0.0",
        "service": "DrawMind API"
    }

# Database stats endpoint
@app.get("/stats")
def stats():
    """
    Returns the total number of drawings and a breakdown by component type.
    """
    try:
        total, component_types = get_database_stats()
        return {
            "total": total,
            "component_types": component_types
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Ingest drawings endpoint
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Uploads and ingests a single engineering drawing.
    Runs Groq Vision AI extraction, embeds result, stores in ChromaDB.

    Args:
        file: Image file (PNG, JPG, WEBP)

    Returns:
        success (bool): Whether ingestion succeeded.
        filename (str): Name of the ingested file.
        component_type (str): Extracted component type.
        material (str): Extracted material if available.
        drawing_number (str): Drawing number if visible.
        scale (str): Scale if visible.
        units (str): Units used in the drawing.
    """
    allowed_extensions = {".png",".jpg",".jpeg",".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPG, JPEG, and WEBP are allowed.")

    try:
        save_path = os.path.join(DRAWINGS_FOLDER, file.filename)
        os.makedirs(DRAWINGS_FOLDER, exist_ok=True)

        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        success, result = ingest_single(save_path, file.filename)

        if success:
            return {
                "success": True,
                "filename": file.filename,
                "component_type": result.get("component_type", "unknown"),
                "material": result.get("material", "unknown"),
                "drawing_number": result.get("drawing_number", "unknown"),
                "scale": result.get("scale", "unknown"),
                "units": result.get("units", "unknown")
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to ingest the drawing.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ask question endpoint
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(request: QuestionRequest):
    """
    Answers a natural language question about the engineering drawings.
    Automatically routes to correct retrieval strategy.

    Args:
        question (str): Natural language question.

    Returns:
        answer (str): Clean natural language answer.
        referenced_drawings (list): Drawing metadata for referenced drawings.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        answer, referenced_metadatas = ask_question(request.question)
        clean_metadatas = []

        for meta in referenced_metadatas:
            raw = json.loads(meta.get("raw_json", "{}"))
            clean_metadatas.append({
                "filename": meta.get("filename"),
                "image_path": meta.get("image_path"),
                "component_type": meta.get("component_type", "unknown"),
                "material": meta.get("material", "unknown"),
                "drawing_number": meta.get("drawing_number", "unknown"),
                "scale": meta.get("scale", "unknown"),
                "units": meta.get("units", "unknown"),
                "dimensions": raw.get("dimensions", []),
                "tolerances": raw.get("tolerances", []),
                "surface_finish": raw.get("surface_finish", "unknown"),
                "notes": raw.get("notes", [])
            })

        return {
            "answer": answer,
            "referenced_drawings": clean_metadatas
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))