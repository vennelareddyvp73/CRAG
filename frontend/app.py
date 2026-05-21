import gradio as gr
import requests
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def upload_document(file):
    """Upload a document to the CRAG backend."""
    if file is None:
        return "No file selected."

    file_path = file.path if hasattr(file, "path") else file
    file_name = file.orig_name if hasattr(file, "orig_name") else os.path.basename(file_path)

    print(f"Uploading file: {file_name} from {file_path}")

    try:
        with open(file_path, "rb") as f:
            response = requests.post(
                f"{API_BASE_URL}/upload",
                files={"file": (file_name, f)}
            )

        if response.status_code == 200:
            data = response.json()
            return f" {data.get('message', 'Document uploaded successfully.')}"
        else:
            return f" Upload failed ({response.status_code}): {response.text}"

    except requests.exceptions.ConnectionError:
        return " Cannot connect to backend. Make sure the FastAPI server is running on port 8000."
    except Exception as e:
        return f" Error: {e}"


def ask_question(question, history):
    """Send a question to the CRAG backend and stream the answer."""
    if not question.strip():
        yield history, ""
        return

    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": ""})
    yield history, ""

    try:

        response = requests.post(
            f"{API_BASE_URL}/query_stream",
            json={"question": question},
            stream=True,
            timeout=60
        )

        if response.status_code == 200:
            try:
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        history[-1]["content"] += chunk
                        yield history, ""
            except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as e:
                history[-1]["content"] += f"\n\n[Connection lost]: {e}"
                yield history, ""
        else:
            history[-1]["content"] = f" Error ({response.status_code}): {response.text}"
            yield history, ""

    except requests.exceptions.ConnectionError:
        history[-1]["content"] = " Cannot connect to backend. Make sure the FastAPI server is running on port 8000."
        yield history, ""
    except Exception as e:
        history[-1]["content"] = f" Unexpected error: {e}"
        yield history, ""


with gr.Blocks(title="CRAG — Corrective RAG") as demo:

    gr.Markdown(
        """
        # CRAG — Corrective Retrieval Augmented Generation
        Upload a document, then ask questions about it.
        The system will retrieve, grade, and generate answers — falling back to web search if needed.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Upload Document")
            file_input = gr.File(
                label="Upload PDF / TXT / DOCX",
                file_types=[".pdf", ".txt", ".docx"]
            )
            upload_btn = gr.Button("Upload & Ingest", variant="primary")
            upload_status = gr.Textbox(
                label="Upload Status",
                interactive=False,
                lines=2
            )

            upload_btn.click(
                fn=upload_document,
                inputs=[file_input],
                outputs=[upload_status]
            )

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask a Question")
            chatbot = gr.Chatbot(
                label="CRAG Chat",
                height=400
            )
            question_input = gr.Textbox(
                placeholder="Ask something about your document...",
                label="Your Question",
                lines=1
            )

            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear Chat")

            submit_btn.click(
                fn=ask_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )

            question_input.submit(
                fn=ask_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )

            clear_btn.click(lambda: ([], ""), outputs=[chatbot, question_input])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
