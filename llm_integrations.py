# ~/ai_agent_system/llm_integrations.py
import json
import os
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import logging
from search_tool import web_search
# Импортируем все необходимые классы сообщений из LangChain Core
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

logger = logging.getLogger(__name__)

load_dotenv()

class LLMIntegration:
    def __init__(self):
        self.hyperbolic_api_keys = {
            # Имена ключей здесь ДОЛЖНЫ совпадать с agent_id, которые используются в database.py и orchestrator.py
            "agent1_hyper0": os.getenv("HYPERBOLIC_API_KEY_0"),
            "agent2_hyper1": os.getenv("HYPERBOLIC_API_KEY_1"),
            "agent3_hyper2": os.getenv("HYPERBOLIC_API_KEY_2"),
            "agent4_hyper3": os.getenv("HYPERBOLIC_API_KEY_3"),
        }
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.nousresearch_api_key = os.getenv("NOUSRESEARCH_API_KEY")

    def get_llm(self, provider, model_name=None, agent_id=None, temperature=0.7, bind_tools=False):
        if provider == "hyperbolic":
            if not agent_id or agent_id not in self.hyperbolic_api_keys:
                raise ValueError(f"Hyperbolic agent_id '{agent_id}' not found in API keys or API key missing.")
            api_key = self.hyperbolic_api_keys[agent_id]
            if not api_key:
                raise ValueError(f"API key for Hyperbolic agent '{agent_id}' is not set (it's None/empty).")

            class HyperbolicLLM:
                def __init__(self, model, api_key, temperature):
                    self.model = model
                    self.api_key = api_key
                    self.temperature = temperature
                    self.url = "https://api.hyperbolic.xyz/v1/chat/completions"

                async def generate(self, messages): # `messages` здесь - это список объектов LangChain BaseMessage
                    api_messages = []
                    for msg in messages:
                        if isinstance(msg, SystemMessage):
                            api_messages.append({"role": "system", "content": msg.content})
                        elif isinstance(msg, HumanMessage):
                            if isinstance(msg.content, str):
                                api_messages.append({"role": "user", "content": msg.content})
                            elif isinstance(msg.content, list):
                                content_parts = []
                                for part in msg.content:
                                    if isinstance(part, dict) and part.get("type") == "text":
                                        content_parts.append(part.get("text"))
                                if content_parts:
                                    api_messages.append({"role": "user", "content": " ".join(content_parts)})
                                else:
                                    logger.warning(f"Unsupported multi-modal content in HumanMessage for HyperbolicLLM: {msg.content}. Using string conversion.")
                                    api_messages.append({"role": "user", "content": str(msg.content)})
                            else:
                                logger.warning(f"Unexpected HumanMessage content type: {type(msg.content)}. Converting to string.")
                                api_messages.append({"role": "user", "content": str(msg.content)})
                        elif isinstance(msg, AIMessage):
                            api_messages.append({"role": "assistant", "content": msg.content})
                        elif isinstance(msg, ToolMessage):
                            api_messages.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content})
                        else:
                            logger.warning(f"Unsupported message type for Hyperbolic API: {type(msg)}. Converting to user message.")
                            api_messages.append({"role": "user", "content": str(msg)})


                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                    data = {
                        "messages": api_messages,
                        "model": self.model,
                        "max_tokens": 1024,
                        "temperature": self.temperature,
                        "top_p": 0.9
                    }
                    try:
                        async with httpx.AsyncClient() as client:
                            response = await client.post(self.url, headers=headers, json=data, timeout=180) # <-- УВЕЛИЧЕН ТАЙМАУТ
                            response.raise_for_status()
                            response_json = response.json()
                            if 'choices' in response_json and response_json['choices']:
                                return response_json['choices'][0]['message']['content']
                            else:
                                logger.error(f"No choices in Hyperbolic API response: {response_json}")
                                return "No response content from Hyperbolic LLM."
                    except httpx.RequestError as e:
                        logger.error(f"Hyperbolic API request failed: {e}")
                        raise
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode JSON from Hyperbolic API: {e}. Response: {response.text}")
                        raise
                    except KeyError as e:
                        logger.error(f"Missing key in Hyperbolic API response: {e}. Response: {response_json}")
                        raise

            return HyperbolicLLM(model_name, api_key, temperature)

        elif provider == "openrouter":
            if not self.openrouter_api_key:
                raise ValueError("OpenRouter API key is not set.")
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature,
                request_timeout=180 # <-- УВЕЛИЧЕН ТАЙМАУТ
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
                temperature=temperature,
                request_timeout=180 # <-- УВЕЛИЧЕН ТАЙМАУТ
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

if __name__ == '__main__':
    import asyncio
    async def test_llms():
        llm_integration = LLMIntegration()
        # Test Hyperbolic LLM
        try:
            print("\nTesting Hyperbolic LLM (Agent 1):")
            hyper_llm = llm_integration.get_llm("hyperbolic", "meta-llama/Meta-Llama-3.1-405B-Instruct", "agent1_hyper0")
            test_messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="What is the capital of France?")
            ]
            hyper_response = await hyper_llm.generate(test_messages)
            print(f"Hyperbolic response: {hyper_response}")
        except Exception as e:
            print(f"Error testing Hyperbolic LLM: {e}")

        # Test OpenRouter LLM with tool binding (Agent 3/4 type)
        try:
            print("\nTesting OpenRouter LLM with tool binding:")
            openrouter_llm_with_tool = llm_integration.get_llm("openrouter", "mistralai/mistral-7b-instruct-v0.2", bind_tools=True)
            print(f"OpenRouter LLM with tool initialized: {openrouter_llm_with_tool.tools}")
            # You would typically test tool calling via AgentExecutor in orchestrator
        except Exception as e:
            print(f"Error testing OpenRouter LLM with tools: {e}")
            
        # Test NousResearch LLM (Agent 6 type)
        try:
            print("\nTesting NousResearch LLM:")
            nous_llm = llm_integration.get_llm("nousresearch", "Nous-Hermes-3.1-Llama-3.1-405B")
            test_messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Who is Alan Turing?")
            ]
            nous_response = await nous_llm.ainvoke(test_messages)
            print(f"NousResearch response: {nous_response.content}")
        except Exception as e:
            print(f"Error testing NousResearch LLM: {e}")

    asyncio.run(test_llms())
