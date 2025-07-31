# 📄 CRAG LangChain Document Question Answering

This repository contains a LangChain-based PDF document question-answering system using Hugging Face models and FAISS vector storage. The system processes documents (such as the CRAG dataset documentation or research papers), enables similarity-based document retrieval, performs document grading for relevance, and generates final answers using an LLM (e.g., LLaMA 3 from Hugging Face).

---

## 🎯 Objective

To build an intelligent PDF question-answering pipeline using:
- **LangChain**
- **HuggingFace LLMs**
- **Sentence Transformers for Embedding**
- **FAISS for vector storage and similarity search**
- **Graph-based conditional flow (LangGraph)**

The pipeline allows you to:
1. Ingest and chunk a PDF.
2. Embed and store in a FAISS vector DB.
3. Retrieve and grade relevant chunks based on user queries.
4. Generate high-quality responses using an LLM.
5. Fallback to web search if document content is insufficient.

---

## 🧠 Architecture Overview

The system performs the following steps:

1. **Embedding**: PDF is chunked and converted to sentence embeddings using `all-MiniLM-L6-v2`.
2. **Storage**: Chunks are stored in a FAISS index.
3. **Retriever**: Fetches top-k relevant documents based on cosine similarity.
4. **Relevance Grading**: A second LLM determines if chunks are semantically related to the query.
5. **Web Search (Fallback)**: If no chunk is relevant, it rewrites the query and runs a web search.
6. **Answer Generation**: Final generation uses prompt + filtered documents with LLaMA 3.2–3B Instruct model.
