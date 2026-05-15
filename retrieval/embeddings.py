import os
from typing import List
import time
import shutil

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from models.embedding_model import get_embedding_model, HuggingFaceEmbeddingProvider


class VectorStoreSingleton:
    """
    Singleton to hold FAISS indexes in memory to prevent redundant disk loads.
    """
    _instance = None
    _indexes = {}  

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorStoreSingleton, cls).__new__(cls)
        return cls._instance

    def get_index(self, folder_path: str, embedding_model):
        if folder_path not in self._indexes:
            self._indexes[folder_path] = FAISS.load_local(
                folder_path,
                embedding_model,
                allow_dangerous_deserialization=True
            )
        return self._indexes[folder_path]

    def set_index(self, folder_path: str, store: FAISS):
        self._indexes[folder_path] = store


class VectorStoreManager:
    """
    Handles embedding model and FAISS operations with Singleton caching.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        provider = HuggingFaceEmbeddingProvider()
        self.embedding_model = get_embedding_model(provider, model_name)
        self.vector_cache = VectorStoreSingleton()

    def build_index(self, chunks: List[Document]) -> FAISS:
        """Create FAISS index from chunks."""
        return FAISS.from_documents(chunks, self.embedding_model)

    def save_index(self, store: FAISS, folder_path: str):
        """Save FAISS index to disk and cache in RAM."""
        store.save_local(folder_path)
        self.vector_cache.set_index(folder_path, store)

    def load_index(self, folder_path: str) -> FAISS:
        """Load FAISS index from RAM if available, else load from disk and cache it."""
        return self.vector_cache.get_index(folder_path, self.embedding_model)

    def get_retriever(self, store: FAISS, k: int = 4):
        """Return retriever from given store."""
        return store.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    try:


        test_chunks = [
            Document(page_content="GANs are feasible for Telugu word generation."),
            Document(page_content="FastAPI provides a high-performance backend.")
        ]

        manager = VectorStoreManager()
        test_folder = "test_faiss_index"

        print("--- 1. Building and Saving Index ---")
        store = manager.build_index(test_chunks)
        manager.save_index(store, test_folder)
        print(f" Index built and saved to '{test_folder}' (and cached in RAM)")

        manager.get_retriever(store, k=1).invoke("warmup")

        print("\n--- 2. First Query (Fetching from RAM Cache) ---")
        start_time = time.time()
        cached_store = manager.load_index(test_folder)
        retriever = manager.get_retriever(cached_store, k=1)
        results = retriever.invoke("What is FastAPI?")
        print(f" Result: {results[0].page_content}")
        print(f" Time taken (RAM cache): {time.time() - start_time:.5f} seconds")

        print("\n--- 3. Force Disk Load (Simulating fresh start) ---")
        manager.vector_cache._indexes.clear()  # manually clear RAM cache
        start_time = time.time()
        disk_store = manager.load_index(test_folder)
        retriever = manager.get_retriever(disk_store, k=1)
        results = retriever.invoke("What is FastAPI?")
        print(f" Result: {results[0].page_content}")
        print(f" Time taken (Disk load): {time.time() - start_time:.5f} seconds")

        # Cleanup
        if os.path.exists(test_folder):
            shutil.rmtree(test_folder)

    except Exception as e:
        print(f" Error: {e}")