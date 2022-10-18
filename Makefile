create-venv:
	python -m venv venv
	python -m pip install --upgrade pip

reqs:
	pip install -r requirements.txt

dev-reqs: reqs
	pip install -r requirements_dev.txt