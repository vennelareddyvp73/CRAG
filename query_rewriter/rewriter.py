from abc import ABC, abstractmethod

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models.llm import get_llm, HuggingFaceProvider


#  Base Strategy
class RewriteStrategy(ABC):
    @abstractmethod
    def rewrite(self, question: str) -> str:
        pass


#  Concrete Strategy (LLM-based rewriting)
class LLMQuestionRewriter(RewriteStrategy):
    def __init__(
        self,
        provider=None,
        model_id: str = "meta-llama/Llama-3.3-70B-Instruct",
        temperature: float = 0
    ):
        if provider is None:
            provider = HuggingFaceProvider()

        self.llm = get_llm(provider, model_id=model_id, temperature=temperature)

        system_prompt = """
You are a question re-writer that converts an input question to a better version that is optimized for web search.
Look at the input and try to reason about the underlying semantic intent / meaning.
Just give only rewritten question. Don't give anything else.
"""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Here is the initial question:\n\n{question}\n\nFormulate an improved question.")
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def rewrite(self, question: str) -> str:
        result = self.chain.invoke({"question": question})
        return result.strip()


#  Context Class
class QuestionRewriter:
    def __init__(self, strategy: RewriteStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: RewriteStrategy):
        self.strategy = strategy

    def rewrite(self, question: str) -> str:
        return self.strategy.rewrite(question)


#  Helper function (for pipeline)
def rewrite_question(
    question: str,
    provider=None,
    model_id: str = "meta-llama/Llama-3.3-70B-Instruct"
) -> str:
    rewriter = QuestionRewriter(LLMQuestionRewriter(provider=provider, model_id=model_id))
    return rewriter.rewrite(question)


#  Test Block
if __name__ == "__main__": 
    try:
        question = "what is bigram ?"

        rewriter = QuestionRewriter(LLMQuestionRewriter())
        improved_question = rewriter.rewrite(question)

        print("Original Question:", question)
        print("\nRewritten Question:", improved_question)

    except Exception as e:
        print(f"Error: {e}")