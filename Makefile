.PHONY: all install lint test format

check: lint test

developer_env:
	pip install -r developer_requirements.txt

install:
	pip install -r requirements.txt

lint:
	isort .
	flake8 --ignore=W605,W503 --exclude learners --exclude tests .
	pylama --skip tests/run_test.py,learners/facmac/learners/facmac_learner_discrete.py,learners/facmac/learners/facmac_learner.py,config/compare_inputs.py .
	pylint *.py
	mypy .

quick_run:
	python main.py -n 5 --n_repeats 2 --n_epochs 20 -o grdC -o bat_dem_agg -o avail_EV_step --rnn_hidden_dim 100

q_learning:
	python main.py -n 20 --n_repeats 5 --n_epochs 30 -o grdC --gamma 0.99 --lr 0.01

facmac:
	python main.py -n 20 --n_repeats 5 --n_epochs 30 -o grdC --gamma 0.99 --lr 0.000001

test:
	pytest tests
