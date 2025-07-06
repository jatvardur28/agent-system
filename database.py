# ~/ai_agent_system/database.py
import sqlite3
import logging

logger = logging.getLogger(__name__)

DATABASE_FILE = 'agents_config.db'

def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            llm_provider TEXT NOT NULL,
            llm_model TEXT,
            system_prompt TEXT
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("Database initialized.")

def add_or_update_agent(agent_id, name, llm_provider, llm_model, system_prompt):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT OR REPLACE INTO agents (id, name, llm_provider, llm_model, system_prompt) VALUES (?, ?, ?, ?, ?)",
                       (agent_id, name, llm_provider, llm_model, system_prompt))
        conn.commit()
        logger.info(f"Agent '{name}' ({agent_id}) added/updated.")
    finally:
        conn.close()

def get_agent_config(agent_id):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, llm_provider, llm_model, system_prompt FROM agents WHERE id = ?", (agent_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return {'id': result[0], 'name': result[1], 'llm_provider': result[2],
                'llm_model': result[3], 'system_prompt': result[4]}
    return None

def get_all_agent_configs():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, llm_provider, llm_model, system_prompt FROM agents")
    results = cursor.fetchall()
    conn.close()
    return [{'id': r[0], 'name': r[1], 'llm_provider': r[2], 'llm_model': r[3], 'system_prompt': r[4]} for r in results]

init_db()
