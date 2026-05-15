from abc import ABC, abstractmethod
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Strategy Interface
class ChunkingStrategy(ABC):
    @abstractmethod
    def split(self, docs: List[Document]) -> List[Document]:
        pass

# Concrete Strategy 
# RecursiveCharacterTextSplitter
class RecursiveChunking(ChunkingStrategy):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)

# Context Class 
class Chunker:
    def __init__(self, strategy: ChunkingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ChunkingStrategy):
        """Allows switching strategies at runtime if needed"""
        self._strategy = strategy

    def execute_chunking(self, docs: List[Document]) -> List[Document]:
        return self._strategy.split(docs)

# test
if __name__ == "__main__":
    # sample = [Document(page_content="This is a long telugu word image research text...")]
    # strategy = RecursiveChunking(chunk_size=25, chunk_overlap=10)

    # chunker = Chunker(strategy)
    # chunks = chunker.execute_chunking(sample)
    
    # print(f"Created {len(chunks)} chunks.")
    # print(chunks)
    pass
    
    