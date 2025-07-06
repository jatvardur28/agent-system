Шаг 3: Запуск системы на VPS
Убедитесь, что вы находитесь в директории ~/ai_agent_system/ и виртуальное окружение активировано.
Generated bash
cd ~/ai_agent_system
source venv/bin/activate
Use code with caution.
Bash
Запустите главного бота:
Generated bash
python3 telegram_bot.py
Use code with caution.
Bash
Если все настроено правильно, вы увидите логи в консоли, и бот начнет отвечать в Telegram.
Запуск в фоновом режиме (рекомендуется для VPS):
Используйте tmux или systemd, как было описано ранее, чтобы обеспечить непрерывную работу.
Использование tmux:
Generated bash
tmux new -s ai_bot_session # Создать новую сессию tmux с именем ai_bot_session
# Внутри сессии tmux:
cd ~/ai_agent_system
source venv/bin/activate
python3 telegram_bot.py
# Чтобы отсоединиться от сессии tmux и оставить её работать: Ctrl+b, затем d
Use code with caution.
Bash
Чтобы снова присоединиться к сессии: tmux attach -t ai_bot_session
Настройка systemd:
Создайте файл юнит-сервиса:
Generated bash
sudo nano /etc/systemd/system/ai_agent_bot.service
Use code with caution.
Bash
Вставьте следующее содержимое (замените your_username_on_vps и /home/your_username_on_vps на ваши реальные данные):
Generated code
[Unit]
Description=AI Agent Telegram Bot
After=network.target

[Service]
User=your_username_on_vps
WorkingDirectory=/home/your_username_on_vps/ai_agent_system
ExecStart=/home/your_username_on_vps/ai_agent_system/venv/bin/python3 telegram_bot.py
Restart=on-failure
EnvironmentFile=/home/your_username_on_vps/ai_agent_system/.env
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
Use code with caution.
Сохраните файл (Ctrl+X, Y, Enter в nano).
Перезагрузите systemd и включите/запустите сервис:
Generated bash
sudo systemctl daemon-reload
sudo systemctl enable ai_agent_bot.service
sudo systemctl start ai_agent_bot.service
Use code with caution.
Bash
Проверьте статус и логи:
Generated bash
sudo systemctl status ai_agent_bot.service
sudo journalctl -u ai_agent_bot.service -f
