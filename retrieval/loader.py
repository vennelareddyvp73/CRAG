import os
from abc import ABC, abstractmethod
from typing import List, Dict, Type

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

## Factory pattern

class BaseLoader(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def load(self) -> List[Document]:
        """Load documents from the specific file path."""
        pass

class PDFLoader(BaseLoader):
    def load(self) -> List[Document]:
        return PyPDFLoader(self.file_path).load()

class TXTLoader(BaseLoader):
    def load(self) -> List[Document]:
        return TextLoader(self.file_path).load()


class DocumentLoaderFactory:
    _loader_map: Dict[str, Type[BaseLoader]] = {
        ".pdf": PDFLoader,
        ".txt": TXTLoader,
    }

    @classmethod
    def get_loader(cls, file_path: str) -> BaseLoader:
        file_extension = os.path.splitext(file_path)[1].lower()
        loader_class = cls._loader_map.get(file_extension)

        if not loader_class:
            raise ValueError(f"Unsupported file format: {file_extension}")

        return loader_class(file_path) 


if __name__ == "__main__":
    # try:
       
    #     test_file_path = os.path.join(os.path.dirname(__file__), "test_document.txt")
    #     loader = DocumentLoaderFactory.get_loader(test_file_path)
    #     documents = loader.load()

    #     print(f"\nSuccessfully loaded {len(documents)} document(s).")
    #     for i, doc in enumerate(documents):
    #         print(f"Document {i+1} Content: {doc.page_content}")


    # except Exception as e:
    #     print(f"Error: {e}")
    pass