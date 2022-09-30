.PHONY: all install lint test format

all: test lint

install:
	pip install -r requirements.txt

lint:
	isort .
	flake8 --ignore=W605,W503 --exclude learners .
	pylama --skip learners .
	pylint *.py
	mypy .

quick_run:
	python main.py -n 5 --n_repeats 2 --n_epochs 20 -o grdC -o bat_dem_agg -o avail_EV_step --rnn_hidden_dim 100


q_learning:
	python main.py -n 20 --n_repeats 5 --n_epochs 30 -o grdC --gamma 0.99 --aggregate_actions True

test:
	pytest tests
