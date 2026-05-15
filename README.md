# Corrective Retrieval-Augmented Generation (CRAG) Pipeline

An end-to-end RAG pipeline that retrieves, grades, and generates answers from your documents — with automatic web search fallback when the document falls short. Built on **LangGraph**, **FastAPI**, and **Gradio**, with real-time token streaming.


---

## Architecture

This project implements **Corrective-RAG** logic orchestrated by **LangGraph**:

1. **Ingestion & Storage**: Documents (PDF / TXT ) are chunked and embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in a **FAISS** vector store on disk with MD5-based caching to skip re-ingestion.
2. **Retrieval**: Fetches the top-K most relevant chunks from FAISS using cosine similarity.
3. **Relevance Grading**: An LLM grades each retrieved chunk to determine whether it *directly answers* the question — not just loosely relates to it.
4. **Correction Engine**:
   - If chunks are **relevant** → proceed to Answer Generation.
   - If chunks are **irrelevant** → rewrite the query and execute a web search via **Tavily** to find external context.
5. **Answer Generation**: Context is passed to **Meta-Llama 3.3 70B** via the HuggingFace Inference API. The answer is streamed token by token to the frontend.

---

## Optimized Performance (Singletons)

The LLM (`meta-llama/Llama-3.3-70B-Instruct`) and embedding model (`all-MiniLM-L6-v2`) are each loaded exactly once per server session using the **Singleton pattern**. FAISS vector stores are cached in RAM after the first disk load, eliminating redundant I/O on repeated queries.

---

## Project Structure

```
CRAG/
├── api/
│   └── main.py                  # FastAPI backend (upload, query, stream endpoints)
├── frontend/
│   └── app.py                   # Gradio UI (file upload + streaming chatbot)
├── graph/
│   └── crag.py                  # LangGraph state machine (CRAG pipeline)
├── models/
│   ├── llm.py                   
│   └── embedding_model.py      
├── retrieval/
│   ├── loader.py                # DocumentLoader(PDF, TXT)
│   ├── chunking.py              
│   ├── embeddings.py            # VectorStore + FAISS 
│   └── pipeline.py              # IngestionPipeline (load → chunk → embed → store)
├── answer_generator/
│   └── answer_gen.py            # RAGGenerator with streaming support
├── query_rewriter/
│   └── rewriter.py              # LLM-based query rewriter for web search
├── relevance_grader/
│   └── grader.py                # LLM relevance grader
├── web_search_tool/
│   └── tool.py                  # Tavily web search 
├── data/                        # Uploaded documents
├── storage/                     # Persisted FAISS vector stores
├── notebook/                    # Experiments
├── pyproject.toml
└── .env
```

---

## Installation & Usage

### 1. Environment Configuration

Create a `.env` file in the project root with your API keys:

```env
HF_TOKEN="your_huggingface_token"
TAVILY_API_KEY="your_tavily_api_key"
```

### 2. Run Locally


**Terminal 1 — Backend (FastAPI):**
```bash
uv run uvicorn api.main:app --port 8000 --reload
```

**Terminal 2 — Frontend (Gradio):**
```bash
python frontend/app.py
```

Open your browser at **http://127.0.0.1:7860**


### 3. Usage

1. Upload a PDF, TXT file using the **Upload Document** panel.
2. Wait for the "Document uploaded and ingested" confirmation.
3. Type your question and click **Ask** — the answer streams back in real time.
4. If the document doesn't contain the answer, the pipeline automatically rewrites your query and falls back to a **Tavily web search**.

---

*Built with `uv`, LangChain, LangGraph, HuggingFace Inference API, Tavily*
