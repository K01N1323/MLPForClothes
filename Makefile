# Настройки
PYTHON  ?= python3
VENV    := .venv
PIP     := $(VENV)/bin/pip
PY      := $(VENV)/bin/python

.PHONY: all venv install run clean help

# По умолчанию просто подготавливаем окружение
all: install

# Создание виртуального окружения
venv: $(VENV)/bin/activate

$(VENV)/bin/activate:
	@echo "==> Создание виртуального окружения..."
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

# Установка зависимостей
install: venv
	@echo "==> Установка зависимостей..."
	$(PIP) install -r requirements.txt

# Запуск основного скрипта
run: install
	@echo "==> Запуск проекта..."
	$(PY) main.py

# Очистка проекта от временных файлов и окружения
clean:
	@echo "==> Очистка..."
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	find . -type f -name "*~" -delete
	find . -type f -name ".DS_Store" -delete

# Справка
help:
	@echo "Использование:"
	@echo "  make         - Создать venv и установить зависимости"
	@echo "  make run     - Запустить main.py (с предварительной установкой)"
	@echo "  make clean   - Удалить venv и кэш Python"
	@echo "  make help    - Показать это сообщение"

