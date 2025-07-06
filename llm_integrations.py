# ~/ai_agent_system/llm_integrations.py
import os
import httpx # Для асинхронных запросов к Hyperbolic
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import logging
from search_tool import web_search # Импортируем наш инструмент

logger = logging.getLogger(__name__)

load_dotenv() # Загружаем переменные окружения

class LLMIntegration:
    def __init__(self):
        self.hyperbolic_api_keys = {
            "hyper0": os.getenv("HYPERBOLIC_API_KEY_0"),
            "hyper1": os.getenv("HYPERBOLIC_API_KEY_1"),
            "hyper2": os.getenv("HYPERBOLIC_API_KEY_2"),
            "hyper3": os.getenv("HYPERBOLIC_API_KEY_3"),
        }
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.nousresearch_api_key = os.getenv("NOUSRESEARCH_API_KEY")

    def get_llm(self, provider, model_name=None, agent_id=None, temperature=0.7, bind_tools=False):
        if provider == "hyperbolic":
            if not agent_id or agent_id not in self.hyperbolic_api_keys:
                raise ValueError(f"Hyperbolic agent_id '{agent_id}' not found or API key missing.")
            api_key = self.hyperbolic_api_keys[agent_id]
            if not api_key:
                raise ValueError(f"API key for Hyperbolic agent '{agent_id}' is not set.")

            # Custom wrapper for Hyperbolic (uses direct httpx requests for async)
            class HyperbolicLLM:
                def __init__(self, model, api_key, temperature):
                    self.model = model
                    self.api_key = api_key
                    self.temperature = temperature
                    self.url = "https://api.hyperbolic.xyz/v1/chat/completions"

                async def generate(self, messages):
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                    data = {
                        "messages": messages,
                        "model": self.model,
                        "max_tokens": 1024, # Adjust as needed
                        "temperature": self.temperature,
                        "top_p": 0.9
                    }
                    try:
                        async with httpx.AsyncClient() as client:
                            response = await client.post(self.url, headers=headers, json=data, timeout=120) # Увеличил таймаут
                            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                            response_json = response.json()
                            return response_json['choices'][0]['message']['content']
                    except httpx.RequestError as e:
                        logger.error(f"Hyperbolic API request failed: {e}")
                        raise

            return HyperbolicLLM(model_name, api_key, temperature)

        elif provider == "openrouter":
            if not self.openrouter_api_key:
                raise ValueError("OpenRouter API key is not set.")
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature
            )
            # Если bind_tools=True, привязываем инструмент web_search к этому LLM
            if bind_tools:
                logger.info(f"Binding web_search tool to OpenRouter LLM: {model_name}")
                # LangChain будет использовать схему web_search для активации внутреннего поиска OpenRouter
                return llm.bind_tools([web_search])
            return llm

        elif provider == "nousresearch":
            if not self.nousresearch_api_key:
                raise ValueError("NousResearch API key is not set.")
            return ChatOpenAI(
                model=model_name,
                openai_api_key=self.nousresearch_api_key,
                # base_url="https://api.nousresearch.com/v1" # Проверьте, есть ли у Nous Research специфический base_url
                temperature=temperature
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

# Пример использования (для тестирования)
if __name__ == '__main__':
    import asyncio
    llm_integration = LLMIntegration()
    async def test_llms():
        try:
            hyper_llm = llm_integration.get_llm("hyperbolic", "meta-llama/Meta-Llama-3.1-405B-Instruct", "hyper0")
            print("Hyperbolic LLM initialized successfully.")
            # Пример вызова Hyperbolic LLM (раскомментируйте для теста)
            # result = await hyper_llm.generate([{"role": "user", "content": "Hello, Hyperbolic!"}])
            # print(f"Hyperbolic response: {result}")

            openrouter_llm_with_tool = llm_integration.get_llm("openrouter", "openrouter/mistralai/mistral-7b-instruct", bind_tools=True)
            print("OpenRouter LLM with tool initialized successfully.")
            
            nous_llm = llm_integration.get_llm("nousresearch", "Nous-Hermes-3.1-Llama-3.1-405B")
            print("NousResearch LLM initialized successfully.")
        except ValueError as e:
            print(f"Error initializing LLMs: {e}")
    
    asyncio.run(test_llms())
