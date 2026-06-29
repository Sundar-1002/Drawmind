# DrawMind
### Engineering Drawing Intelligence System

DrawMind is a multimodal RAG system that transforms unstructured engineering drawing images into an intelligent, searchable knowledge base. Vision AI automatically extracts metadata from each drawing and makes the entire collection queryable through natural language.

---

## The Problem

Companies store millions of engineering drawings as image files. These drawings contain critical information including component types, materials, dimensions, tolerances, and manufacturing specifications. But because they are images, none of that information is searchable or queryable without opening each file manually.

DrawMind solves this.

---

## Live Demo

The system is fully deployed and accessible:

- **Frontend:** Streamlit Cloud
- **Backend API:** Railway
- **Vectors:** Pinecone (cloud, persistent)
- **Images:** Cloudinary (cloud, persistent)

---

## How It Works

```
Engineering Drawing (image)
        ↓
Groq Vision AI reads the drawing
Extracts: component type, material, dimensions,
tolerances, surface finish, drawing number, scale
        ↓
HuggingFace SentenceTransformer embeds extracted text
        ↓
Pinecone stores vectors and metadata permanently
Image stored in Cloudinary with public URL
        ↓
User asks a natural language question
        ↓
System routes question to correct retrieval strategy
        ↓
Groq LLM generates grounded answer
        ↓
Referenced drawings shown with images
```

---

## Smart Question Routing

DrawMind does not use one size fits all semantic search. It routes each question to the most accurate retrieval strategy:

| Question Type | Example | Strategy |
|---|---|---|
| Global overview | "What components are available?" | Compact summary of entire database |
| Filename specific | "Give me image of 8.jpg" | Direct fetch by ID from Pinecone |
| Component type | "Show me all gear drawings" | Metadata filter, no semantic search |
| Semantic | "Find shafts longer than 500mm" | Vector similarity search |

---

## Features

- Automatic metadata extraction from engineering drawing images using Groq Vision AI
- Natural language Q&A across all drawings
- Smart four level question routing for accurate results
- Real time ingestion: upload new drawings from the UI, instantly searchable
- Honest edge case handling: never hallucinates results when fewer drawings exist than requested
- LLM controlled image display: shows referenced drawings only when relevant
- Conversational memory within a session for follow up questions
- Fully hosted with persistent cloud storage

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Vision AI | Groq Llama 4 Scout 17B | Reads and extracts data from drawing images |
| LLM | Groq Llama 3.1 8B | Answers questions from retrieved context |
| Vector Database | Pinecone | Persistent cloud vector storage |
| Image Storage | Cloudinary | Persistent public image URLs |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Converts text to semantic vectors |
| Backend API | FastAPI | REST API layer |
| Frontend | Streamlit | User interface |

---

## Project Structure

```
DrawMind/
    core/
        __init__.py
        ingest.py       # Vision AI extraction and Pinecone storage
        search.py       # Question routing and retrieval logic
    api.py              # FastAPI REST endpoints
    app.py              # Streamlit frontend
    requirements.txt
    render.yaml
    .env                # API keys (not committed)
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Check if API is running |
| GET | /stats | Total drawings and component breakdown |
| POST | /ingest | Upload and ingest a new drawing |
| POST | /ask | Ask a natural language question |

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sundar-1002/DrawMind
cd DrawMind
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac or Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=drawmind
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_api_key
CLOUDINARY_API_SECRET=your_cloudinary_api_secret
API_URL=http://localhost:8000
```

### 5. Ingest drawings

Place engineering drawing images in the `drawings/` folder then run:

```bash
python core/ingest.py
```

### 6. Run locally

Open two terminals:

**Terminal 1 (FastAPI):**
```bash
uvicorn api:app --reload --port 8000
```

**Terminal 2 (Streamlit):**
```bash
streamlit run app.py
```

---

## Example Questions

| Type | Question |
|---|---|
| Overview | "What components are available in the database?" |
| Component | "Show me all gear drawings" |
| Material | "Which drawing has alloy steel material?" |
| Dimension | "Find shafts longer than 500mm" |
| Analytical | "Which drawing has the largest dimensions?" |
| Audit | "Which drawings are missing material specification?" |
| Filename | "Give me image of 8.jpg" |
| Edge case | "Give me 20 bolt drawings" |

---

## Architecture Decisions

**Why Vision AI instead of PDF text extraction?**
Engineering drawings are image based. Dimensions, symbols, and text are drawn as part of the image rather than embedded as digital text. Vision AI reads the drawing visually exactly as a human engineer would.

**Why Pinecone instead of local ChromaDB?**
ChromaDB stores data locally and loses everything when the container resets. Pinecone is a managed cloud vector database. Data persists permanently regardless of server restarts or redeployments.

**Why direct metadata filtering over semantic search for component questions?**
Semantic search retrieves the top N most similar vectors which may miss some components or include irrelevant ones when one component type dominates the database. Direct metadata filtering returns every matching drawing with 100 percent accuracy.

**Why LLM controlled image display?**
Instead of always showing referenced drawings or never showing them, the LLM decides based on the question type. Analytical questions get text only answers. Show or display questions get images. This makes the UI feel intelligent rather than mechanical.

---

## Production Roadmap

| Component | Current | Production Path |
|---|---|---|
| Vector Database | Pinecone Starter | Pinecone Standard with namespaces |
| Image Storage | Cloudinary Free | Cloudinary or AWS S3 |
| Ingestion | Manual upload via UI | Airflow automated pipeline |
| Observability | Railway logs | OpenTelemetry, Grafana, Prometheus |
| Quality Control | Manual testing | Automated LLM-as-judge, RAGAS evaluation |
| Frontend | Streamlit | React or Next.js |

---

## Related Projects

**DocuMind** — LLM based research agent using LangChain, LangGraph, ChromaDB, FastAPI, and Strawberry GraphQL. Scored 18/20 on RAGAS evaluation.
GitHub: https://github.com/Sundar-1002/DocuMind

---

## Author

**Sundararajan Srinivasan**
MSc Data and Computational Science, University College Dublin (First Class Honours)

LinkedIn: https://www.linkedin.com/in/sundararajan2001/
GitHub: https://github.com/Sundar-1002/