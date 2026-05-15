import os
import hashlib
import time
import shutil


from retrieval.loader import DocumentLoaderFactory
from retrieval.chunking import Chunker, RecursiveChunking
from retrieval.embeddings import VectorStoreManager


class IngestionPipeline:
    def __init__(self, storage_path: str = "storage/vectorstores"):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

        self.chunker = Chunker(RecursiveChunking())
        self.vector_manager = VectorStoreManager()

    def _generate_doc_id(self, file_path: str) -> str:
        """
        Generate unique hash for the document
        """
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        return hashlib.md5(file_bytes).hexdigest()

    def run(self, file_path: str):
        """
        Full ingestion pipeline:
        Load → Chunk → Embed → Store (with caching)
        """

        doc_id = self._generate_doc_id(file_path)
        index_path = os.path.join(self.storage_path, doc_id)


        if os.path.exists(index_path):
            print(" Loading existing vector store...")
            return self.vector_manager.load_index(index_path)

        print(" Processing new document...")

        # 1. Load
        loader = DocumentLoaderFactory.get_loader(file_path)
        docs = loader.load()

        # 2. Chunk
        chunks = self.chunker.execute_chunking(docs)

        # 3. Embed
        store = self.vector_manager.build_index(chunks)

        # 4. Save
        self.vector_manager.save_index(store, index_path)

        return store


# 🔹 Test Block
if __name__ == "__main__":
    try:

        # Create a self-contained test file
        test_file = os.path.join(os.path.dirname(__file__), "pipeline_test.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("This is a test document for the end-to-end ingestion pipeline. It will be chunked, embedded, and stored.")

        pipeline = IngestionPipeline(storage_path="test_pipeline_store")

        print("--- Running Pipeline (First Time) ---")
        start = time.time()
        store = pipeline.run(test_file)
        print(f" Time taken: {time.time() - start:.3f}s")

        print("\n--- Running Pipeline (Second Time / Cached) ---")
        start = time.time()
        store2 = pipeline.run(test_file)
        print(f" Time taken: {time.time() - start:.3f}s")

        retriever = store.as_retriever(search_kwargs={"k": 1})
        query = "What is this document about?"
        results = retriever.invoke(query)

        print(f"\n Query: {query}")
        print(f" Result: {results[0].page_content}")

        # Cleanup test artifacts
        if os.path.exists("test_pipeline_store"):
            shutil.rmtree("test_pipeline_store")
        if os.path.exists(test_file):
            os.remove(test_file)

    except Exception as e:
        print(f" Error: {e}")