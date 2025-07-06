sudo nano /etc/systemd/system/telegram-bot.service

Вставьте следующее содержимое:
[Unit]
Description=AI Agent Telegram Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/ai_agent_system
Environment="PATH=/root/ai_agent_system/venv/bin"
ExecStart=/root/ai_agent_system/venv/bin/python3 /root/ai_agent_system/telegram_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target


Затем:
# Перезагрузите systemd
sudo systemctl daemon-reload

# Включите автозапуск
sudo systemctl enable telegram-bot.service

# Запустите службу
sudo systemctl start telegram-bot.service

# Проверьте статус
sudo systemctl status telegram-bot.service

# Посмотреть логи
sudo journalctl -u telegram-bot.service -f
