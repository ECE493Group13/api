import json
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import pytest
from flask.testing import FlaskClient

from api.database import (
    DatasetModel,
    DatasetPaperModel,
    FilterTaskModel,
    PaperModel,
    TrainTaskModel,
    UserModel,
    db,
)
from api.workers import trainer
from api.workers.trainer import TrainWorker
from api.workers.worker import WorkerRunner


@pytest.fixture()
def dataset(authorized_user: UserModel, papers: list[PaperModel]):
    dataset_ = DatasetModel(num_papers=0, name="")

    task = FilterTaskModel(
        user=authorized_user,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        keywords="back pain",
        dataset=dataset_,
    )

    db.session.add_all([dataset_, task])
    db.session.flush()
    db.session.add(DatasetPaperModel(dataset_id=dataset_.id, dkey=papers[0].dkey))
    db.session.commit()
    return dataset_


@pytest.fixture()
def train_task(dataset: DatasetModel, hparams: dict):
    task = TrainTaskModel(
        hparams=json.dumps(hparams), user=dataset.task.user, dataset=dataset
    )
    db.session.add(task)
    db.session.commit()
    return task


@pytest.fixture()
def hparams():
    return {
        "embedding_size": 200,
        "epochs_to_train": 1,
        "learning_rate": 0.025,
        "num_neg_samples": 25,
        "batch_size": 500,
        "concurrent_steps": 12,
        "window_size": 5,
        "min_count": 1,
        "subsample": 1e-3,
    }


@pytest.fixture()
def data_root():
    with TemporaryDirectory() as tempdir:
        yield Path(tempdir)


class TestTrainTask:
    def test_post(
        self,
        client: FlaskClient,
        dataset: DatasetModel,
        hparams: dict,
        auth_headers: dict,
    ):
        """
        Posting to /train-task creates and returns a task from the given dataset
        """
        response = client.post(
            "/train-task",
            json={"dataset_id": dataset.id, "hparams": hparams},
            headers=auth_headers,
        )
        assert response.status_code == HTTPStatus.CREATED
        assert response.json["hparams"] == json.dumps(hparams, sort_keys=True)
        assert response.json["created"] is not None
        assert response.json["start_time"] is None
        assert response.json["end_time"] is None
        assert response.json["is_complete"] is False
        assert response.json["is_error"] is False

        task = (
            db.session.query(TrainTaskModel)
            .filter_by(id=response.json["id"])
            .one_or_none()
        )
        assert task is not None
        assert task.dataset == dataset

    def test_list(
        self,
        client: FlaskClient,
        train_task: TrainTaskModel,
        auth_headers: dict,
    ):
        """
        Can list train tasks and filter by dataset
        """
        response = client.get("/train-task", headers=auth_headers)
        assert response.status_code == HTTPStatus.OK
        assert len(response.json) == 1

        response = client.get(
            f"/train-task?dataset_id={train_task.dataset_id}", headers=auth_headers
        )
        assert response.status_code == HTTPStatus.OK
        assert len(response.json) == 1

        response = client.get(
            f"/train-task?dataset_id={train_task.dataset_id+1}", headers=auth_headers
        )
        assert response.status_code == HTTPStatus.OK
        assert len(response.json) == 0

    def test_get(
        self, client: FlaskClient, train_task: TrainTaskModel, auth_headers: dict
    ):
        """
        Can get a single train task by id
        """
        response = client.get(f"/train-task/{train_task.id}", headers=auth_headers)
        assert response.status_code == HTTPStatus.OK
        assert response.is_json

        response = client.get(f"/train-task/{train_task.id+1}", headers=auth_headers)
        assert response.status_code == HTTPStatus.NOT_FOUND

    def test_suggest_hparams(self, client: FlaskClient, auth_headers: dict):
        """
        Can get suggested hyperparameters for the user to modify
        """
        response = client.get("/train-task/suggest-hparams", headers=auth_headers)
        assert response.status_code == HTTPStatus.OK
        expected_keys = {
            "embedding_size",
            "epochs_to_train",
            "learning_rate",
            "num_neg_samples",
            "batch_size",
            "concurrent_steps",
            "window_size",
            "min_count",
            "subsample",
        }
        assert set(response.json.keys()) == expected_keys
        assert response.json["embedding_size"] == 200


class TestTrainWorker:
    def test_write_corpus(self, dataset: DatasetModel):
        """
        Ngrams should be written to the corpus file
        """
        with NamedTemporaryFile() as file:
            trainer.write_corpus(db.session, dataset, Path(file.name))
            db.session.commit()
            assert file.read().decode().splitlines()[:2] == [
                "aliqua\t2",
                "incididunt\t7",
            ]

    def test_generate_visualization(self, dataset: DatasetModel, hparams: dict):
        """
        Should generate TSNe raw data for embeddings
        """
        task = TrainTaskModel(
            hparams=json.dumps(hparams), user=dataset.task.user, dataset=dataset
        )
        db.session.add(task)
        db.session.commit()

        with NamedTemporaryFile("wb") as file:
            file.write(b"2 1\nhello 0.1\nworld 0.2\n")
            file.flush()
            visualization = trainer.generate_visualization(Path(file.name))

        assert visualization is not None
        visualization = json.loads(visualization)
        assert isinstance(visualization["labels"], list)
        assert isinstance(visualization["x"], list)
        assert isinstance(visualization["y"], list)

    def test_model_save(self, dataset: DatasetModel, hparams: dict):
        """
        Should save model to the db
        """
        task = TrainTaskModel(
            hparams=json.dumps(hparams), user=dataset.task.user, dataset=dataset
        )
        db.session.add(task)
        db.session.commit()

        with NamedTemporaryFile("wb") as file:
            file.write(b"2 1\nhello 0.1\nworld 0.2\n")
            file.flush()
            visualization = trainer.generate_visualization(Path(file.name))
            trainer.save_model(db.session, task, Path(file.name), visualization)

            assert task.model is not None
            assert task.model.visualization is not None
            assert task.model.embeddings_filename is not None

    def test_execute(self, dataset: DatasetModel, hparams: dict, data_root: Path):
        """
        Should successfully train, producing a model
        """
        task = TrainTaskModel(
            hparams=json.dumps(hparams), user=dataset.task.user, dataset=dataset
        )
        db.session.add(task)
        db.session.commit()

        WorkerRunner(TrainWorker(data_root))._tick(db.session)  # pylint: disable=W0212
        db.session.commit()

        assert task.model is not None
        assert task.is_complete is True
        assert task.is_error is False
