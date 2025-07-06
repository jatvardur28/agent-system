# ~/ai_agent_system/orchestrator.py
import json
import logging
import asyncio
from typing import Dict, Any, List

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler

from llm_integrations import LLMIntegration
from search_tool import ALL_TOOLS # Список инструментов (только web_search)
import database

logger = logging.getLogger(__name__)

llm_integration = LLMIntegration()

class TelegramCallbackHandler(BaseCallbackHandler):
    """
    Коллбэк-обработчик для LangChain, который отправляет информацию о действиях агента в Telegram.
    """
    def __init__(self, chat_id: int, send_message_callback):
        super().__init__()
        self.chat_id = chat_id
        self.send_message_callback = send_message_callback

    async def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
        log_message = action.log
        if len(log_message) > 500:
            log_message = log_message[:497] + "..."
        # Экранируем потенциальные символы Markdown в логе, чтобы избежать BadRequest
        escaped_log_message = log_message.replace('_', r'\_').replace('*', r'\*').replace('`', r'\`')
        await self.send_message_callback(self.chat_id, f"➡️ _{escaped_log_message}_", parse_mode='Markdown')

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        tool_name = serialized.get("name", "Unknown Tool")
        await self.send_message_callback(self.chat_id, f"🛠️ *Использую инструмент* `{tool_name}`: `{input_str}`", parse_mode='Markdown')

    async def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        truncated_output = (output[:500] + '...') if len(output) > 500 else output
        # Экранируем потенциальные символы Markdown
        escaped_output = truncated_output.replace('_', r'\_').replace('*', r'\*').replace('`', r'\`')
        await self.send_message_callback(self.chat_id, f"✅ *Инструмент завершил работу.* Результат (обрезано): `{escaped_output}`", parse_mode='Markdown')

    async def on_agent_finish(self, finish: Any, **kwargs: Any) -> Any:
        pass


async def create_agent_from_config(agent_id: str, telegram_callback_handler: TelegramCallbackHandler = None):
    """
    Создает экземпляр агента (LLM Chain или AgentExecutor) на основе его конфигурации из БД.
    """
    config = database.get_agent_config(agent_id)
    if not config:
        logger.error(f"Agent configuration for '{agent_id}' not found.")
        return None

    llm = llm_integration.get_llm(
        provider=config['llm_provider'],
        model_name=config['llm_model'],
        agent_id=config['id'] if config['llm_provider'] == 'hyperbolic' else None,
        bind_tools=(agent_id in ["agent3_hyper2", "agent4_hyper3"]) 
    )

    base_agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", config['system_prompt']),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    
    if agent_id in ["agent3_hyper2", "agent4_hyper3"]:
        agent = create_tool_calling_agent(llm, ALL_TOOLS, base_agent_prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS,
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[telegram_callback_handler] if telegram_callback_handler else None
        )
        return executor
    else:
        class SimpleChainWrapper:
            def __init__(self, llm_instance, system_prompt):
                self.llm_instance = llm_instance
                self.system_prompt = system_prompt
            
            async def ainvoke(self, input_data: Dict[str, Any]):
                user_message = input_data.get('input', '')
                
                # Формируем список сообщений для LLM
                messages_for_llm: List[Any] = [SystemMessage(content=self.system_prompt), HumanMessage(content=user_message)]

                if hasattr(self.llm_instance, 'generate'): # For our custom HyperbolicLLM
                    response_content = await self.llm_instance.generate(messages_for_llm)
                    return {"output": response_content}
                else: # For LangChain ChatOpenAI LLM and Nous LLM (теперь OpenRouter)
                    # Возвращаемся к прямому списку сообщений, что должно быть стабильно с зафиксированными версиями
                    response = await self.llm_instance.ainvoke(messages_for_llm) # <-- ИЗМЕНЕНО: Прямо список BaseMessage
                    return {"output": response.content}

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
        escaped_error_msg = str(e).replace('_', r'\_').replace('*', r'\*').replace('`', r'\`')
        await send_message_callback(chat_id, f"⚠️ **Ошибка Агента #1:** {escaped_error_msg}", parse_mode='Markdown')
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
            clean_json_string = orchestration_plan_raw.strip()
            if clean_json_string.startswith("```json"):
                json_start_tag = "```json\n"
                json_end_tag = "\n```"
                json_start_index = clean_json_string.find(json_start_tag)
                json_end_index = clean_json_string.rfind(json_end_tag)

                if json_start_index != -1 and json_end_index != -1 and json_end_index > (json_start_index + len(json_start_tag)):
                    clean_json_string = clean_json_string[json_start_index + len(json_start_tag) : json_end_index].strip()
                else:
                    logger.warning("Markdown JSON block found but could not be cleanly parsed. Attempting raw JSON load.")
            
            orchestration_plan = json.loads(clean_json_string)

            agent3_task = orchestration_plan.get('agent3_task')
            agent4_task = orchestration_plan.get('agent4_task')

            if not agent3_task or not agent4_task:
                raise ValueError("Parsed plan is missing 'agent3_task' or 'agent4_task'. Check Agent #2's output format.")

            await send_message_callback(chat_id, f"📋 **Агент #2 завершил.** План сформирован для Агентов #3 и #4.", parse_mode='Markdown')
            logger.info(f"Agent 2 output plan: {orchestration_plan}")

        except json.JSONDecodeError as e:
            escaped_error_msg = str(e).replace('_', r'\_').replace('*', r'\*').replace('`', r'\`')
            await send_message_callback(chat_id, f"⚠️ **Ошибка парсинга плана Агента #2:** {escaped_error_msg}\nRaw output: ```{orchestration_plan_raw}```", parse_mode='Markdown')
            logger.error(f"Agent 2 JSON parsing error: {e}, Raw output: {orchestration_plan_raw}")
            return
        except ValueError as e:
            escaped_error_msg = str(e).replace('_', r'\_').replace('*', r'\*').replace('`', r'\`')
            await send_message_callback(chat_id, f"⚠️ **Ошибка структуры плана Агента #2:** {escaped_error_msg}", parse_mode='Markdown')
            logger.error(f"Agent 2 plan structure error: {e}")
            return

    except Exception as e:
        escaped_error_msg = str(e).replace('_', r'\_').replace('*', r'\*').replace('`', r'\`')
        await send_message_callback(chat_id, f"⚠️ **Ошибка Агента #2:** {escaped_error_msg}", parse_mode='Markdown')
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
            combined_input = (
                f"### Ваша задача и инструкции: ###\n"
                f"{task_config['system_prompt']}\n\n"
                f"### Запрос для выполнения: ###\n"
                f"{task_config['instructional_query']}"
            )
            
            result = await executor.ainvoke({"input": combined_input})
            await send_message_callback(chat_id, f"✅ **{agent_label} завершил работу.**", parse_mode='Markdown')
            return result.get('output', f"Не удалось получить результат от {agent_label}.")
        except Exception as e:
            escaped_error_msg = str(e).replace('_', r'\_').replace('*', r'\*').replace('`', r'\`')
            await send_message_callback(chat_id, f"⚠️ **Ошибка {agent_label}:** {escaped_error_msg}", parse_mode='Markdown')
            logger.exception(f"{agent_label} failed.")
            return f"Error: {e}"

    # Запускаем в параллель
    try:
        results = await asyncio.gather(
            run_research_agent(agent3_executor, agent3_task, "Агент #3"),
            run_research_agent(agent4_executor, agent4_task, "Агент #4"),
            return_exceptions=True
        )
        agent3_res, agent4_res = results

        if isinstance(agent3_res, Exception):
            await send_message_callback(chat_id, f"❌ **Агент #3 потерпел сбой:** {agent3_res}", parse_mode='Markdown')
            agent3_res = "Результат Агента #3 недоступен из-за ошибки."
        if isinstance(agent4_res, Exception):
            await send_message_callback(chat_id, f"❌ **Агент #4 потерпел сбой:** {agent4_res}", parse_mode='Markdown')
            agent4_res = "Результат Агента #4 недоступен из-за ошибки."

    except Exception as e:
        escaped_error_msg = str(e).replace('_', r'\_').replace('*', r'\*').replace('`', r'\`')
        await send_message_callback(chat_id, f"⚠️ **Ошибка при параллельном выполнении Агентов #3/#4:** {escaped_error_msg}", parse_mode='Markdown')
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
        escaped_error_msg = str(e).replace('_', r'\_').replace('*', r'\*').replace('`', r'\`')
        await send_message_callback(chat_id, f"⚠️ **Ошибка Агента #6:** {escaped_error_msg}", parse_mode='Markdown')
        logger.exception("Agent 6 failed.")
        
        try:
            await send_message_callback(chat_id, "⚠️ Ошибка при создании финального отчета (подробнее в логах сервера).", parse_mode=None)
        except Exception as e_plain:
            logger.error(f"Failed to send plain error message to Telegram: {e_plain}")
            
        return

    await send_message_callback(chat_id, "\n✨ **Процесс завершен!**", parse_mode='Markdown')
