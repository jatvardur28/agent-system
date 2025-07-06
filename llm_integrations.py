# ~/ai_agent_system/llm_integrations.py
import os
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import logging
from search_tool import web_search

logger = logging.getLogger(__name__)

load_dotenv()

class LLMIntegration:
    def __init__(self):
        self.hyperbolic_api_keys = {
            # Имена ключей здесь ДОЛЖНЫ совпадать с agent_id, которые используются в database.py и orchestrator.py
            "agent1_hyper0": os.getenv("HYPERBOLIC_API_KEY_0"),
            "agent2_hyper1": os.getenv("HYPERBOLIC_API_KEY_1"),
            "agent3_hyper2": os.getenv("HYPERBOLIC_API_KEY_2"), # Агент 3 теперь OpenRouter, но его ключ Hyperbolic все равно здесь
            "agent4_hyper3": os.getenv("HYPERBOLIC_API_KEY_3"), # Агент 4 теперь OpenRouter, но его ключ Hyperbolic все равно здесь
        }
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.nousresearch_api_key = os.getenv("NOUSRESEARCH_API_KEY")

    def get_llm(self, provider, model_name=None, agent_id=None, temperature=0.7, bind_tools=False):
        if provider == "hyperbolic":
            # Проверяем agent_id здесь, чтобы выбрать правильный ключ из словаря
            if not agent_id or agent_id not in self.hyperbolic_api_keys:
                raise ValueError(f"Hyperbolic agent_id '{agent_id}' not found in API keys or API key missing.")
            api_key = self.hyperbolic_api_keys[agent_id]
            if not api_key:
                raise ValueError(f"API key for Hyperbolic agent '{agent_id}' is not set (it's None/empty).")

            # ... остальная часть метода get_llm для hyperbolic LLM (без изменений) ...
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
                        "max_tokens": 1024,
                        "temperature": self.temperature,
                        "top_p": 0.9
                    }
                    try:
                        async with httpx.AsyncClient() as client:
                            response = await client.post(self.url, headers=headers, json=data, timeout=120)
                            response.raise_for_status()
                            response_json = response.json()
                            return response_json['choices'][0]['message']['content']
                    except httpx.RequestError as e:
                        logger.error(f"Hyperbolic API request failed: {e}")
                        raise

            return HyperbolicLLM(model_name, api_key, temperature)

        # ... остальная часть метода get_llm (openrouter, nousresearch) без изменений ...
        elif provider == "openrouter":
            if not self.openrouter_api_key:
                raise ValueError("OpenRouter API key is not set.")
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature
            )
            if bind_tools:
                logger.info(f"Binding web_search tool to OpenRouter LLM: {model_name}")
                return llm.bind_tools([web_search])
            return llm

        elif provider == "nousresearch":
            if not self.nousresearch_api_key:
                raise ValueError("NousResearch API key is not set.")
            return ChatOpenAI(
                model=model_name,
                openai_api_key=self.nousresearch_api_key,
                temperature=temperature
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

# ... (конец файла) ...
