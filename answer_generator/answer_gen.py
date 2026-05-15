from abc import ABC, abstractmethod
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from models.llm import get_llm, HuggingFaceProvider


#  Base Strategy
class GenerationStrategy(ABC):
    @abstractmethod
    def generate(self, docs: List[Document], question: str) -> str:
        pass

    @abstractmethod
    def stream(self, docs: List[Document], question: str):
        pass


#  Concrete Strategy (RAG-based generation)
class RAGGenerator(GenerationStrategy):
    def __init__(self):
       
        provider = HuggingFaceProvider()
        self.llm = get_llm(
            provider,
            model_id="meta-llama/Llama-3.3-70B-Instruct",
            temperature=0
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a helpful AI assistant.
Use the provided context to answer the question.
If the answer is not in the context, say you don't know.
Keep the answer concise and clear."""
            ),
            ("human", "Context:\n\n{context}\n\nQuestion: {question}")
        ])

       
        self.chain = (self.prompt | self.llm | StrOutputParser()).with_config({"tags": ["final_answer"]})

    def format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def generate(self, docs: List[Document], question: str) -> str:
        context = self.format_docs(docs)

        response = self.chain.invoke({
            "context": context,
            "question": question
        })

        return response

    def stream(self, docs: List[Document], question: str):
        context = self.format_docs(docs)

        for chunk in self.chain.stream({
            "context": context,
            "question": question
        }):
            yield chunk


#  Context Class
class AnswerGenerator:
    def __init__(self, strategy: GenerationStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: GenerationStrategy):
        self.strategy = strategy

    def generate(self, docs: List[Document], question: str) -> str:
        return self.strategy.generate(docs, question)

    def stream(self, docs: List[Document], question: str):
        return self.strategy.stream(docs, question)


#  Helper function (for pipeline)
def generate_answer(docs: List[Document], question: str) -> str:
    generator = AnswerGenerator(RAGGenerator())
    return generator.generate(docs, question)


def stream_answer(docs: List[Document], question: str):
    generator = AnswerGenerator(RAGGenerator())
    yield from generator.stream(docs, question)


#  Test Block
if __name__ == "__main__":
    # try:
    #     docs = [
    #         Document(page_content="FastAPI is a modern Python web framework."),
    #         Document(page_content="It is used for building APIs efficiently.")
    #     ]

    #     question = "What is FastAPI?"

    #     generator = AnswerGenerator(RAGGenerator())
    #     answer = generator.generate(docs, question)

    #     print("Question:", question)
    #     print("\nAnswer:\n", answer)

    # except Exception as e:
    #     print(f"Error: {e}")
    pass