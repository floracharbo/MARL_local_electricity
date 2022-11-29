.PHONY: all install lint test format

developer_env:
	pip install -r config_files/environments/developer_requirements.txt

install:
	pip install -r requirements.txt

lint:
	isort src
	flake8 --ignore=W605,W503 --exclude learners --exclude tests --max-line-length=100 src
	pylama --skip src/tests/run_test.py,src/learners/facmac --ignore=E501 src
	find src -type f -not -path "src/tests/*" -not -path "src/learners/facmac/*" -name "*.py" | xargs pylint --disable=W0201,E1101,E0401
	mypy --show-error-codes --exclude src/learners/facmac --disable-error-src import --disable-error-src attr-defined src --disable-error-src no-member --disable-error-src duplicate-src


quick_run:
	python main.py -n 5 --n_repeats 2 --n_epochs 20 -o grdC -o bat_dem_agg -o avail_car_step --rnn_hidden_dim 100

q_learning:
	python main.py -n 20 --n_repeats 5 --n_epochs 30 -o grdC --gamma 0.99 --lr 0.01

facmac:
	python main.py -n 20 --n_repeats 5 --n_epochs 30 -o grdC --gamma 0.99 --lr 0.000001

test:
	python -m pytest src/tests
