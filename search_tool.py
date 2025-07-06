# ~/ai_agent_system/search_tool.py
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Определяем инструмент web_search, который LLM на OpenRouter будет использовать.
# Сама логика выполнения поиска находится на стороне OpenRouter,
# этот класс лишь предоставляет его схему для LangChain.
@tool("web_search")
def web_search(query: str) -> str:
    """
    Performs a comprehensive internet search to retrieve up-to-date information.
    Input should be a clear, concise search query relevant to the information needed.
    Returns the search results summary.
    """
    # Эта функция на самом деле не будет вызываться напрямую в Python.
    # Она служит для генерации схемы инструмента, которую LangChain передаст LLM.
    # LLM (через OpenRouter) будет использовать эту схему, чтобы понять,
    # как вызвать свой внутренний инструмент поиска.
    logger.warning("The web_search tool's Python function is a placeholder and should not be directly executed.")
    return "This tool is handled directly by the LLM's underlying platform (OpenRouter)."

# Список всех инструментов, которые могут быть использованы агентами
ALL_TOOLS = [web_search]

if __name__ == '__main__':
    # Пример использования для демонстрации схемы инструмента
    print(web_search.name)
    print(web_search.description)
    print(web_search.args)
