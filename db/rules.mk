include db/secrets.mk

CONTAINER_NAME=postgres
PORT=5433
DATA_MOUNT=$(shell pwd)/db/data/
DB_URL=postgres://postgres:$(PASSWORD)@localhost:$(PORT)/postgres

db-start:
	sudo docker run \
		--publish 127.0.0.1:$(PORT):5432 \
		--volume $(DATA_MOUNT):/var/lib/postgresql/data \
		--name $(CONTAINER_NAME) \
		--env POSTGRES_PASSWORD=$(PASSWORD) \
		--detach \
		postgres

db-stop:
	sudo docker stop $(CONTAINER_NAME)
	sudo docker rm $(CONTAINER_NAME)

db-init:
	./db/init-data.bash $(DB_URL)

psql:
	psql $(DB_URL)

,phony: test-db-init
test-db-init:
	sudo docker run \
		--publish 127.0.0.1:5434:5432 \
		--volume $(shell pwd)/db/test-data/:/var/lib/postgresql/data \
		--name testdb \
		--env POSTGRES_PASSWORD=test_password \
		--detach \
		postgres

.phony: test-db-start
test-db-start:
	sudo docker start testdb

.phony: test-db-stop
test-db-stop:
	sudo docker stop testdb

.phony: test-db-quit
test-db-quit: test-db-stop
	sudo docker rm testdb

.phony: test-psql
test-psql:
	psql postgres://postgres:test_password@localhost:5434/postgres
