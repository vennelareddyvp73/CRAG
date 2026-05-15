import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

class WebSearchStrategy(ABC):
    @abstractmethod
    def search(self, query: str) -> Any:
        pass

class TavilySearchStrategy(WebSearchStrategy):
    def __init__(self, max_results: int = 3):
        self.tool = TavilySearch(max_results=max_results)

    def search(self, query: str) -> List[Dict[str, Any]]:
        return self.tool.invoke({"query": query})

class WebSearchTool:
    def __init__(self, strategy: WebSearchStrategy):
        self._strategy = strategy

    def execute_search(self, query: str) -> Any:
        return self._strategy.search(query)

if __name__ == "__main__":
    try:
        search_context = WebSearchTool(TavilySearchStrategy(max_results=2))
        query = "What is FastAPI?"
        results = search_context.execute_search(query)

        print(f"Query: {query}\n" + "-"*30)

        if isinstance(results, list):
            for i, res in enumerate(results, 1):
                content = res.get("content", "No content available")
                url = res.get("url", "No URL provided")
                print(f"{i}. [Source: {url}]")
                print(f"   {content[:200]}...\n")
        else:
            print(results)

    except Exception as e:
        print(f"An error occurred: {e}")