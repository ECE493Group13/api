include db/rules.mk

.phony: serve
serve:
	DB_PASSWORD=$(PASSWORD) MAIL_PASSWORD=$(MAIL_PASSWORD) FLASK_ENV=development flask run --port 4433

.phony: openapi
openapi:
	flask openapi write openapi.json

.phony: test
test:
	pytest tests

.phony: setup
setup:
	python -m venv venv
	venv/bin/pip install -e .[dev]
