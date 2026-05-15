from abc import ABC, abstractmethod
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models.llm import get_llm, HuggingFaceProvider


class GradingStrategy(ABC):
    @abstractmethod
    def grade(self, document: str, question: str) -> str:
        pass


#  Concrete Strategy (LLM-based grading)
class LLMRelevanceGrader(GradingStrategy):
    def __init__(self):

        provider = HuggingFaceProvider()
        self.llm = get_llm(
            provider,
            model_id="meta-llama/Llama-3.3-70B-Instruct",
            temperature=0
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
             """You are a grader assessing whether a retrieved document can directly answer a user question.

Only say "yes" if the document contains specific information that directly answers the question.
Say "no" if the document is only loosely related, tangentially mentions the topic, or does not contain enough information to answer the question.

Reply with ONLY the single word "yes" or "no" — no explanation, no punctuation."""
            ),
            ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}")
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def grade(self, document: str, question: str) -> str:
        result = self.chain.invoke({
            "document": document,
            "question": question
        })
        return result.strip().lower()


#  Context Class
class RelevanceGrader:
    def __init__(self, strategy: GradingStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: GradingStrategy):
        self.strategy = strategy

    def grade(self, document: str, question: str) -> str:
        return self.strategy.grade(document, question)

    def grade_documents(self, documents: List[str], question: str) -> List[str]:
        return [self.strategy.grade(doc, question) for doc in documents]


#  Helper function (used in pipeline)
def filter_relevant_docs(docs, question: str) -> List:
    """
    Takes LangChain Documents → returns only relevant ones
    """
    grader = RelevanceGrader(LLMRelevanceGrader())

    relevant_docs = []

    for doc in docs:
        result = grader.grade(doc.page_content, question)

        first_word = result.split()[0].strip(".,!?").lower() if result.strip() else "no"

        if first_word == "yes":
            relevant_docs.append(doc)

    return relevant_docs


#  Test Block
if __name__ == "__main__":
    # try:
    #     sample_docs = [
    #         "FastAPI is a modern web framework for building APIs with Python.",
    #         "GANs are used for generating images.",
    #         "The capital of France is Paris."
    #     ]

    #     question = "What is FastAPI?"

    #     grader = RelevanceGrader(LLMRelevanceGrader())

    #     print("Testing Relevance Grader...\n")

    #     for doc in sample_docs:
    #         result = grader.grade(doc, question)
    #         print(f"Doc: {doc}")
    #         print(f"Relevant?: {result}\n")

    # except Exception as e:
    #     print(f"Error: {e}")
    pass