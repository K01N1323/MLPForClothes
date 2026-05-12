PYTHON  ?= python3
VENV    := .venv
PIP     := $(VENV)/bin/pip
PY      := $(VENV)/bin/python

.PHONY: all venv install run clean

all: run

# Создать виртуальное окружение
venv: $(VENV)/bin/activate

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

# Установить зависимости
install: venv
	$(PIP) install -r requirements.txt

# Запустить модель
run: install
	$(PY) model.py

# Удалить виртуальное окружение
clean:
	rm -rf $(VENV)

