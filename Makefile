.PHONY: all install lint test format

all: lint test

install:
	pip install -r requirements.txt

lint:
	isort .
	flake8 --ignore=W605,W503 --exclude learners .
	pylama --skip learners .
	pylint *.py
	mypy .

quick_run:
	python main.py -n 5 --n_repeats 2 --n_epochs 20 -o grdC --o avail_EV_step --rnn_hidden_dim 100

test:
	pytest tests
