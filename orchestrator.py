import json
import logging
import asyncio
from typing import Dict, Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

from llm_integrations import LLMIntegration
from search_tool import ALL_TOOLS # Список инструментов (только web_search)
import database

logger = logging.getLogger(__name__)

llm_integration = LLMIntegration()

class TelegramCallbackHandler:
    """
    Коллбэк-обработчик для LangChain, который отправляет информацию о действиях агента в Telegram.
    """
    def __init__(self, chat_id: int, send_message_callback):
        self.chat_id = chat_id
        self.send_message_callback = send_message_callback

    async def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
        await self.send_message_callback(self.chat_id, f"➡️ _{action.log}_", parse_mode='Markdown')

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        tool_name = serialized.get("name", "Unknown Tool")
        await self.send_message_callback(self.chat_id, f"🛠️ *Использую инструмент* `{tool_name}`: `{input_str}`", parse_mode='Markdown')

    async def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        # Обрезаем вывод, чтобы не спамить и не превышать лимиты Telegram
        truncated_output = (output[:500] + '...') if len(output) > 500 else output
        await self.send_message_callback(self.chat_id, f"✅ *Инструмент завершил работу.* Результат (обрезано): `{truncated_output}`", parse_mode='Markdown')

    async def on_agent_finish(self, finish: Any, **kwargs: Any) -> Any:
        pass # Этот коллбэк для внутренних мыслей агента, не всегда нужно выводить в ТГ


async def create_agent_from_config(agent_id: str, telegram_callback_handler: TelegramCallbackHandler = None):
    """
    Создает экземпляр агента (LLM Chain или AgentExecutor) на основе его конфигурации из БД.
    """
    config = database.get_agent_config(agent_id)
    if not config:
        logger.error(f"Agent configuration for '{agent_id}' not found.")
        return None

    # Создаем LLM на основе конфигурации.
    # Для Агентов 3 и 4 (которые теперь OpenRouter) и которые будут AgentExecutor'ами,
    # мы привязываем инструменты (web_search) к их LLM.
    llm = llm_integration.get_llm(
        provider=config['llm_provider'],
        model_name=config['llm_model'],
        agent_id=config['id'] if config['llm_provider'] == 'hyperbolic' else None, # Передаем agent_id только для Hyperbolic
        bind_tools=(agent_id in ["agent3_hyper2", "agent4_hyper3"]) # Привязываем инструменты к LLM Агентов 3 и 4
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", config['system_prompt']),
            MessagesPlaceholder("chat_history", optional=True), # Для сохранения истории, если потребуется
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"), # Для внутренних мыслей агента в AgentExecutor
        ]
    )
    
    # Агенты 3 и 4 теперь являются AgentExecutor'ами, т.к. их LLM (OpenRouter) поддерживает tool_calling.
    if agent_id in ["agent3_hyper2", "agent4_hyper3"]:
        # Создаем AgentExecutor, который будет использовать LLM с привязанными инструментами
        agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS, # Передаем список доступных инструментов
            verbose=True, # Включаем логирование в консоль
            handle_parsing_errors=True,
            callbacks=[telegram_callback_handler] if telegram_callback_handler else None
        )
        return executor
    else:
        # Для агентов, которые просто генерируют текст без использования LangChain AgentExecutor
        # (Агент 1, Агент 2, Агент 6)
        # А также для Агента 5, который является просто LLM, используемым как движок для поиска.
        class SimpleChainWrapper:
            def __init__(self, llm_instance, system_prompt):
                self.llm_instance = llm_instance
                self.system_prompt = system_prompt
            
            async def ainvoke(self, input_data: Dict[str, Any]):
                user_message = input_data.get('input', '')
                messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=user_message)]

                # Если это наш кастомный HyperbolicLLM, у него есть метод generate
                if hasattr(self.llm_instance, 'generate'):
                    response_content = await self.llm_instance.generate(
                        [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_message}]
                    )
                    return {"output": response_content}
                # Иначе, это LangChain ChatOpenAI LLM
                else:
                    response = await self.llm_instance.ainvoke(messages)
                    return {"output": response.content} # LangChain ChatModel возвращает Content (str)

        return SimpleChainWrapper(llm, config['system_prompt'])


async def run_full_agent_process(user_query: str, chat_id: int, send_message_callback):
    """
    Оркестрирует полный процесс работы агентов: от получения запроса до отправки финального отчета.
    """
    telegram_callback_handler = TelegramCallbackHandler(chat_id, send_message_callback)

    await send_message_callback(chat_id, "🚀 **Инициирую процесс поиска и анализа...**\n\n", parse_mode='Markdown')

    # --- Шаг 1: Агент №1 (Промпт-трансформер) ---
    await send_message_callback(chat_id, "🤖 **Агент #1 (Промпт-трансформер)**: Преобразую ваш запрос...", parse_mode='Markdown')
    agent1 = await create_agent_from_config("agent1_hyper0", telegram_callback_handler)
    if not agent1:
        await send_message_callback(chat_id, "❌ Ошибка: Агент #1 не найден или не настроен.")
        return

    try:
        a1_result = await agent1.ainvoke({"input": user_query})
        refined_query = a1_result.get('output', "Не удалось уточнить запрос.")
        await send_message_callback(chat_id, f"📝 **Агент #1 завершил.** Уточненный запрос:\n```\n{refined_query}\n```", parse_mode='Markdown')
    except Exception as e:
        await send_message_callback(chat_id, f"⚠️ **Ошибка Агента #1:** {e}")
        logger.exception("Agent 1 failed.")
        return

    # --- Шаг 2: Агент №2 (Оркестратор) ---
    await send_message_callback(chat_id, "\n🤖 **Агент #2 (Оркестратор)**: Планирую задачи для исследователей...", parse_mode='Markdown')
    agent2 = await create_agent_from_config("agent2_hyper1", telegram_callback_handler)
    if not agent2:
        await send_message_callback(chat_id, "❌ Ошибка: Агент #2 не найден или не настроен.")
        return
    
    try:
        a2_result = await agent2.ainvoke({"input": refined_query})
        orchestration_plan_raw = a2_result.get('output', "Не удалось получить план оркестрации.")
        
        try:
            orchestration_plan = json.loads(orchestration_plan_raw)
            agent3_task = orchestration_plan.get('agent3_task')
            agent4_task = orchestration_plan.get('agent4_task')

            if not agent3_task or not agent4_task:
                raise ValueError("Parsed plan is missing 'agent3_task' or 'agent4_task'. Check Agent #2's output format.")

            await send_message_callback(chat_id, f"📋 **Агент #2 завершил.** План сформирован для Агентов #3 и #4.", parse_mode='Markdown')
            logger.info(f"Agent 2 output plan: {orchestration_plan}")

        except json.JSONDecodeError as e:
            await send_message_callback(chat_id, f"⚠️ **Ошибка парсинга плана Агента #2:** Ожидался JSON, но получен некорректный формат. {e}\nRaw output: ```{orchestration_plan_raw}```", parse_mode='Markdown')
            logger.error(f"Agent 2 JSON parsing error: {e}, Raw output: {orchestration_plan_raw}")
            return
        except ValueError as e:
            await send_message_callback(chat_id, f"⚠️ **Ошибка структуры плана Агента #2:** {e}", parse_mode='Markdown')
            logger.error(f"Agent 2 plan structure error: {e}")
            return

    except Exception as e:
        await send_message_callback(chat_id, f"⚠️ **Ошибка Агента #2:** {e}", parse_mode='Markdown')
        logger.exception("Agent 2 failed.")
        return

    # --- Шаг 3 & 4: Агенты №3 и №4 (Исследователи) в параллель ---
    await send_message_callback(chat_id, "\n🔄 **Агенты #3 и #4 (Исследователи)**: Запускаю параллельный поиск...", parse_mode='Markdown')

    agent3_executor = await create_agent_from_config("agent3_hyper2", telegram_callback_handler)
    agent4_executor = await create_agent_from_config("agent4_hyper3", telegram_callback_handler)

    if not agent3_executor or not agent4_executor:
        await send_message_callback(chat_id, "❌ Ошибка: Один из исследовательских агентов не найден/не настроен.")
        return

    async def run_research_agent(executor, task_config, agent_label):
        """Вспомогательная функция для запуска агентов-исследователей."""
        await send_message_callback(chat_id, f"🔍 **{agent_label}** начинает исследование...", parse_mode='Markdown')
        try:
            # Для AgentExecutor, мы можем динамически обновлять системный промпт
            # (предполагая, что промпт в database.py является шаблоном)
            dynamic_prompt_messages = [
                SystemMessage(content=task_config['system_prompt']),
                MessagesPlaceholder("chat_history", optional=True),
                HumanMessage(content="{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
            executor.agent.prompt = ChatPromptTemplate.from_messages(dynamic_prompt_messages)
            
            result = await executor.ainvoke({"input": task_config['instructional_query']})
            await send_message_callback(chat_id, f"✅ **{agent_label} завершил работу.**", parse_mode='Markdown')
            return result.get('output', f"Не удалось получить результат от {agent_label}.")
        except Exception as e:
            await send_message_callback(chat_id, f"⚠️ **Ошибка {agent_label}:** {e}", parse_mode='Markdown')
            logger.exception(f"{agent_label} failed.")
            return f"Error: {e}"

    # Запускаем в параллель
    try:
        results = await asyncio.gather(
            run_research_agent(agent3_executor, agent3_task, "Агент #3"),
            run_research_agent(agent4_executor, agent4_task, "Агент #4"),
            return_exceptions=True # Чтобы не упасть, если один из них выдаст ошибку
        )
        agent3_res, agent4_res = results

        if isinstance(agent3_res, Exception):
            await send_message_callback(chat_id, f"❌ **Агент #3 потерпел сбой:** {agent3_res}", parse_mode='Markdown')
            agent3_res = "Результат Агента #3 недоступен из-за ошибки."
        if isinstance(agent4_res, Exception):
            await send_message_callback(chat_id, f"❌ **Агент #4 потерпел сбой:** {agent4_res}", parse_mode='Markdown')
            agent4_res = "Результат Агента #4 недоступен из-за ошибки."

    except Exception as e:
        await send_message_callback(chat_id, f"⚠️ **Ошибка при параллельном выполнении Агентов #3/#4:** {e}", parse_mode='Markdown')
        logger.exception("Parallel execution of Agents 3/4 failed.")
        return

    # --- Шаг 6: Агент №6 (Финальный Аналитик) ---
    await send_message_callback(chat_id, "\n🧠 **Агент #6 (Финальный Аналитик)**: Объединяю и синтезирую результаты...", parse_mode='Markdown')
    agent6 = await create_agent_from_config("agent6_nous0", telegram_callback_handler)
    if not agent6:
        await send_message_callback(chat_id, "❌ Ошибка: Агент #6 не найден или не настроен.")
        return

    final_analysis_input = (
        f"Оригинальный запрос пользователя: {user_query}\n\n"
        f"Результаты от Агента #3:\n{agent3_res}\n\n"
        f"Результаты от Агента #4:\n{agent4_res}\n\n"
        "Объедини и синтезируй эти результаты в единый, структурированный и компетентный отчет."
    )

    try:
        a6_result = await agent6.ainvoke({"input": final_analysis_input})
        final_report = a6_result.get('output', "Не удалось получить финальный отчет.")
        await send_message_callback(chat_id, "✅ **Финальный отчет готов!**", parse_mode='Markdown')
        await send_message_callback(chat_id, final_report, parse_mode='Markdown')
    except Exception as e:
        await send_message_callback(chat_id, f"⚠️ **Ошибка Агента #6:** {e}", parse_mode='Markdown')
        logger.exception("Agent 6 failed.")
        return

    await send_message_callback(chat_id, "\n✨ **Процесс завершен!**", parse_mode='Markdown')
