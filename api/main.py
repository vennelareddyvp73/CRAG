import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from retrieval.pipeline import IngestionPipeline
from graph.crag import app as crag_app


app = FastAPI(title="CRAG API")


UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


_active_doc_path: dict = {"path": None}


@app.get("/")
def health_check():
    return {"status": "ok", "message": "CRAG API is running."}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF or TXT).
    The pipeline will ingest and embed it automatically.
    Subsequent queries will use this document.
    """
    allowed_extensions = (".pdf", ".txt", ".docx")
    if not file.filename.endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    try:
        pipeline = IngestionPipeline()
        pipeline.run(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    _active_doc_path["path"] = file_path

    return {
        "status": "success",
        "filename": file.filename,
        "message": "Document uploaded and ingested. Ready for queries."
    }


class QueryRequest(BaseModel):
    question: str
    doc_path: str = None   


@app.post("/query")
async def query(req: QueryRequest):
    """
    Run the full CRAG pipeline for a given question.
    Uses the last uploaded document unless doc_path is explicitly provided.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    doc_path = req.doc_path or _active_doc_path["path"]

    if not doc_path:
        raise HTTPException(
            status_code=400,
            detail="No document available. Please upload a document first via POST /upload."
        )

    if not os.path.exists(doc_path):
        raise HTTPException(
            status_code=404,
            detail=f"Document not found at path: {doc_path}"
        )

    try:
        result = crag_app.invoke({
            "question": req.question,
            "doc_path": doc_path,
            "documents": [],
            "filtered_docs": [],
            "answer": ""
        })

        return {
            "status": "success",
            "question": req.question,
            "answer": result.get("answer", "No answer found.")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query_stream")
async def query_stream(req: QueryRequest):
    """
    Stream the CRAG pipeline answer token by token.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    doc_path = req.doc_path or _active_doc_path["path"]

    if not doc_path:
        raise HTTPException(
            status_code=400,
            detail="No document available. Please upload a document first."
        )

    async def event_generator():
        try:
            initial_state = {
                "question": req.question,
                "doc_path": doc_path,
                "documents": [],
                "filtered_docs": [],
                "answer": ""
            }

            yield " "

            async for event in crag_app.astream_events(initial_state, version="v2"):
                kind = event["event"]
                tags = event.get("tags", [])

                if kind == "on_chat_model_stream" and "final_answer" in tags:
                    content = event["data"]["chunk"].content
                    if content:
                        yield content
        except Exception as e:
            print(f"Streaming Error: {e}")
            yield f"\n [Streaming Error]: {str(e)}"

    return StreamingResponse(event_generator(), media_type="text/plain")
