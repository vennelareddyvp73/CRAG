from pathlib import Path


PROJECT_STRUCTURE = [

    "api/main.py",

    "frontend/app.py",
    "graph/crag.py",
    "models/llm.py",
    "models/embedding_model.py",


    "retrieval/chunking.py",
    "retrieval/embeddings.py",
    "retrieval/loader.py",
    "retrieval/pipeline.py",

    "answer_generator/answer_gen.py",

    "query_rewriter/rewriter.py",

    "relevance_grader/grader.py",
    


    "web_search_tool/tool.py",

    "notebook/CRAG.ipynb",


]


def create_file(path: str):
    file_path = Path(path)
    
    if file_path.name == ".gitkeep":
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()
        print(f"Created Directory: {file_path.parent}/")
        return

    if not file_path.exists():
   
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create empty file
        file_path.touch()
        print(f"Created File: {file_path}")
    else:
        print(f"Skipped (Already Exists): {file_path}")


def generate():
   
    base_path = Path.cwd()

    for file_path in PROJECT_STRUCTURE:
        create_file(str(base_path / file_path))
        
    print("Project structure generated")


if __name__ == "__main__":
    generate()
