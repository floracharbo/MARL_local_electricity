.PHONY: all install lint test format

all: format lint test

install:
	pip install -r requirements.txt

lint:
	flake8 --ignore=W605,W503 --exclude learners .
	pylama --skip learners .
	pylint *.py
	mypy .

test:
	pytest tests

format:
	isort .
