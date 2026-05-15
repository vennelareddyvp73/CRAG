from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Singleton pattern


class EmbeddingProvider(ABC):
    @abstractmethod
    def create_embedding_model(self, model_id: str):
        pass


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    def create_embedding_model(self, model_id: str):
        hf_token = os.getenv("HF_TOKEN")

        if hf_token is None:
            raise ValueError("HF_TOKEN not found in .env")

        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

        return HuggingFaceEmbeddings(
            model_name=model_id,
            model_kwargs={"token": hf_token}
        )


class EmbeddingSingleton:
    _instance = None
    _model = None
    _model_id = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingSingleton, cls).__new__(cls)
        return cls._instance

    def get_model(self, provider: EmbeddingProvider, model_id: str):
        if self._model is None or self._model_id != model_id:
            self._model = provider.create_embedding_model(model_id)
            self._model_id = model_id
        return self._model


def get_embedding_model(provider: EmbeddingProvider, model_id: str):
    return EmbeddingSingleton().get_model(provider, model_id)


if __name__ == "__main__":
    try:
        hf_provider = HuggingFaceEmbeddingProvider()

        embedding_model = get_embedding_model(
            hf_provider,
            model_id="sentence-transformers/all-MiniLM-L6-v2"
        )

        query = "What is the capital of France?"
        embedding = embedding_model.embed_query(query)

        print(f"Successfully generated embedding for query: '{query}'")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 5 vector values: {embedding[:5]}")

    except Exception as e:
        print(f"Error: {e}")
