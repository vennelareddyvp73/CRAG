from typing import List, TypedDict
from langchain_core.documents import Document

from retrieval.pipeline import IngestionPipeline
from retrieval.embeddings import VectorStoreManager

from relevance_grader.grader import filter_relevant_docs
from answer_generator.answer_gen import generate_answer
from query_rewriter.rewriter import rewrite_question
from web_search_tool.tool import WebSearchTool, TavilySearchStrategy

from langgraph.graph import StateGraph, END


#  Graph State
class CRAGState(TypedDict):
    question: str
    doc_path: str           
    documents: List[Document]
    filtered_docs: List[Document]
    answer: str


#  RETRIEVE
def retrieve(state: CRAGState):
    question = state["question"]
    doc_path = state["doc_path"]

    pipeline = IngestionPipeline()
    store = pipeline.run(doc_path)

    vs = VectorStoreManager()
    retriever = vs.get_retriever(store)
    docs = retriever.invoke(question)

    return {"documents": docs}


#  GRADE
def grade_documents(state: CRAGState):
    docs = state["documents"]
    question = state["question"]

    filtered = filter_relevant_docs(docs, question)

    return {"filtered_docs": filtered}


#  DECISION
def decide(state: CRAGState):
    if len(state["filtered_docs"]) > 0:
        return "generate"
    else:
        return "rewrite"


#  GENERATE
def generate(state: CRAGState):
    docs = state["filtered_docs"]
    question = state["question"]

    answer = generate_answer(docs, question)

    return {"answer": answer}


#  REWRITE
def rewrite(state: CRAGState):
    question = state["question"]
    new_question = rewrite_question(question)

    return {"question": new_question}


#  WEB SEARCH
def web_search(state: CRAGState):
    question = state["question"]

    tool = WebSearchTool(TavilySearchStrategy())
    results = tool.execute_search(question)

    docs = []


    raw = results.get("results", []) if isinstance(results, dict) else results
    for r in raw:
        content = r.get("content", "") if isinstance(r, dict) else str(r)
        if content:
            docs.append(Document(page_content=content))

    return {"filtered_docs": docs}


graph = StateGraph(CRAGState)

graph.add_node("retrieve", retrieve)
graph.add_node("grade", grade_documents)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite)
graph.add_node("web_search", web_search)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "grade")

graph.add_conditional_edges(
    "grade",
    decide,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

graph.add_edge("rewrite", "web_search")
graph.add_edge("web_search", "generate")

graph.add_edge("generate", END)

app = graph.compile()


#  TEST
if __name__ == "__main__":
    import os

    # Create a small test document at the project root
    test_file = "C:/Users/venne/Videos/Screenshots/Documents/CRAG/data/sample.pdf"
    question = "What does autoregressive language models mean?"

    result = app.invoke({
        "question": question,
        "doc_path": test_file,
        "documents": [],
        "filtered_docs": [],
        "answer": ""
    })

    print("\nFinal Answer:\n", result["answer"])

  