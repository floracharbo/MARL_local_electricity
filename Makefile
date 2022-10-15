.PHONY: all install lint test format

check: test

developer_env:
	pip install -r config_files/environments/developer_requirements.txt

install:
	pip install -r requirements.txt

lint:
	isort code
	flake8 --ignore=W605,W503 --exclude learners --exclude tests --max-line-length=100 code
	pylama --ignore code/tests/run_test.py,code/learners/facmac --ignore=E501 code
	find code -type f -not -path "./tests/*" -name "*.py" | xargs pylint --ignore=tests/* --disable=W0201
	mypy --show-error-codes --exclude code/learners/facmac --disable-error-code import --disable-error-code attr-defined code --disable-error-code no-member --disable-error-code duplicate-code

quick_run:
	python code/main.py -n 5 --n_repeats 2 --n_epochs 20 -o grdC -o bat_dem_agg -o avail_EV_step --rnn_hidden_dim 100

q_learning:
	python code/main.py -n 20 --n_repeats 5 --n_epochs 30 -o grdC --gamma 0.99 --lr 0.01

facmac:
	python code/main.py -n 20 --n_repeats 5 --n_epochs 30 -o grdC --gamma 0.99 --lr 0.000001

test:
	pytest code/tests
