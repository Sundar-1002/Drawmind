# DrawMind
### Engineering Drawing Intelligence System

DrawMind is a multimodal RAG system that transforms unstructured engineering drawing images into an intelligent, searchable knowledge base. Instead of storing drawings as inaccessible image files, DrawMind uses Vision AI to automatically extract metadata and makes every drawing queryable through natural language.

---

## The Problem

Companies store millions of engineering drawings as image files. These drawings contain critical information: component types, materials, dimensions, tolerances, and specifications. But without a way to search them, this data is essentially dark and inaccessible. A procurement manager cannot simply ask "show me all stainless steel components above 100mm diameter" across thousands of drawings.

DrawMind solves this.

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
ChromaDB stores vectors and metadata
        ↓
User asks a natural language question
        ↓
System routes question to correct retrieval strategy
        ↓
Groq LLM generates a grounded answer
        ↓
Referenced drawings shown alongside the answer
```

---

## Features

- **Automatic metadata extraction** from engineering drawing images using Groq Vision AI (Llama 4 Scout)
- **Natural language Q&A** across all drawings using Groq LLM (Llama 3.1)
- **Smart question routing**: component specific questions use direct metadata filtering; general questions use semantic search; database overview questions fetch all drawings
- **Real time ingestion**: upload new drawings directly from the UI and they become instantly searchable
- **Honest edge case handling**: system accurately reports when fewer drawings exist than requested, never hallucinating results
- **Component type detection**: automatically classifies gears, shafts, brackets, housings, fasteners, flanges, and more

---

## Tech Stack

| Component | Technology |
|---|---|
| Vision AI for extraction | Groq (Llama 4 Scout 17B) |
| LLM for Q&A | Groq (Llama 3.1 8B Instant) |
| Vector database | ChromaDB (persistent) |
| Text embeddings | HuggingFace sentence-transformers (all-MiniLM-L6-v2) |
| Backend functions | Python |
| Frontend | Streamlit |

---

## Project Structure

```
DrawMind/
    drawings/           # engineering drawing image files
    chroma_db/          # persistent vector database
    core/
        __init__.py
        ingest.py       # vision extraction and ChromaDB storage
        search.py       # Q&A, routing, and retrieval logic
    app.py              # Streamlit UI
    requirements.txt
    .env                # API keys (not committed)
```

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

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at: https://console.groq.com

### 5. Add engineering drawings

Place your engineering drawing images (PNG, JPG, WEBP) inside the `drawings/` folder.

### 6. Ingest drawings

```bash
python core/ingest.py
```

This processes every drawing through Groq Vision and stores extracted metadata in ChromaDB.

### 7. Run the app

```bash
streamlit run app.py
```

---

## Usage

### Asking Questions

Type any natural language question in the input box:

| Question Type | Example |
|---|---|
| Component search | "Show me all gear drawings" |
| Material search | "Which drawing has stainless steel material" |
| Database overview | "What components are available in the database" |
| Missing data audit | "Which drawings are missing material specification" |
| Drawing specific | "What are the tolerances in drawing 9.jpg" |
| Edge case | "Give me 20 bolt drawings" |

### Uploading New Drawings

Use the **Add New Drawings** section in the sidebar to upload new drawing images. They are automatically ingested and become searchable immediately without restarting the app.

---

## Architecture Decisions

**Why Vision AI instead of PDF text extraction?**
Engineering drawings are image based. Dimensions, symbols, and text are drawn as part of the image rather than embedded as digital text. Standard PDF text extraction returns nothing useful. Vision AI reads the drawing visually, exactly as a human engineer would.

**Why direct metadata filtering over semantic search for component questions?**
When a user asks "show me gear drawings", semantic search retrieves the top N most similar vectors which may include irrelevant components if gears dominate the database. Direct metadata filtering queries ChromaDB for exact component type matches, giving precise and reliable results.

**Why compact context for global questions?**
Passing full JSON for 30 plus drawings exceeds LLM token limits. For overview questions, only key fields (filename, component type, material, drawing number, scale, units) are passed as compact summaries, reducing token usage by approximately 80 percent while retaining all information needed to answer the question.

**Why LLM based component extraction from questions?**
Hardcoded keyword matching breaks when users ask about component types not in the predefined list. Using a small LLM call to extract the component type from the question handles synonyms, plurals, and novel component types automatically without any code changes.

---

## Roadmap

- Structured numerical filtering (find gears with diameter greater than 100mm)
- Similarity search by image (upload an unknown drawing and find similar ones)
- Automated ingestion pipeline using Airflow for continuous data freshness
- Migration to Pinecone for enterprise scale vector storage
- Export search results as CSV for procurement workflows
- Multi language support for international engineering drawings

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
