# ~/ai_agent_system/telegram_bot.py
import os
import logging
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes
)
from telegram.constants import ParseMode # <-- ДОБАВЛЕНО для явного использования ParseMode

# Импортируем модули
import database
import agent_definitions # Для инициализации агентов в БД
import orchestrator

load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Команды бота ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение."""
    await update.message.reply_text(
        "Привет! Я агентская система для поиска и анализа информации. "
        "Отправь мне свой запрос, чтобы начать.\n\n"
        "Я буду кратко транслировать весь путь выполнения задачи.",
        parse_mode=ParseMode.MARKDOWN_V2 # Убедимся, что Markdown всегда используется
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает текстовые сообщения как запросы на поиск и анализ."""
    user_query = update.message.text
    chat_id = update.message.chat_id

    logger.info(f"Received query from chat_id {chat_id}: {user_query}")

    # Запускаем оркестратор в отдельной задаче, чтобы бот не зависал
    # Передаем context.bot.send_message как callback для отправки промежуточных сообщений
    asyncio.create_task(
        orchestrator.run_full_agent_process(user_query, chat_id, context.bot.send_message)
    )
    # Первое сообщение пользователю, что запрос принят
    await update.message.reply_text("Ваш запрос принят в работу. Пожалуйста, ожидайте, это может занять некоторое время...")


# --- Основная функция запуска бота ---
def main() -> None:
    """Запускает бота."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN не установлен в .env")
        print("Ошибка: TELEGRAM_BOT_TOKEN не установлен. Пожалуйста, проверьте файл .env")
        return

    # ИЗМЕНЕНИЕ ЗДЕСЬ: Увеличиваем таймаут для HTTP-запросов к Telegram API
    # read_timeout - таймаут на чтение ответа сервера
    # write_timeout - таймаут на отправку запроса
    application = (
        Application.builder()
        .token(token)
        .http_version("1.1") # Рекомендуется для python-telegram-bot с httpx
        .arbitrary_callback_data(True) # Если используете inline кнопки, это может быть полезно
        .read_timeout(180) # <-- УВЕЛИЧЕНО до 3 минут
        .write_timeout(180) # <-- УВЕЛИЧЕНО до 3 минут
        .build()
    )
    
    # Добавляем обработчики команд
    application.add_handler(CommandHandler("start", start))

    # Добавляем обработчик для текстовых сообщений (кроме команд)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен. Ожидаю сообщений...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    # Убедимся, что база данных и агенты инициализированы перед запуском бота
    database.init_db() # Инициализирует БД
    agent_definitions.define_agents() # Заполняет БД конфигурациями агентов
    main()
