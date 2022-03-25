import json
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock

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


@pytest.fixture()
def dataset(authorized_user: UserModel, papers: list[PaperModel]):
    dataset_ = DatasetModel()

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


class TestTrainer:
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

    def test_read_embeddings(self, dataset: DatasetModel, hparams: dict):
        """
        Should create a new trained model for the embeddings
        """
        task = TrainTaskModel(
            hparams=json.dumps(hparams), user=dataset.task.user, dataset=dataset
        )
        db.session.add(task)
        db.session.commit()

        with NamedTemporaryFile("wb") as file:
            file.write(b"hello world")
            file.flush()
            trainer.read_embeddings(db.session, task, Path(file.name))
        db.session.commit()
        assert task.model is not None
        assert task.model.data == b"2 1\nhello 0.1\nworld 0.2\n"
        assert task.model.visualization is not None

    def test_tick(self, dataset: DatasetModel, hparams: dict):
        """
        Should successfully train, producing a model and marking task as done
        """
        task = TrainTaskModel(
            hparams=json.dumps(hparams), user=dataset.task.user, dataset=dataset
        )
        db.session.add(task)
        db.session.commit()

        trainer.tick(db.session)

        assert task.model is not None
        assert task.end_time is not None

    def test_tick_failed(self, dataset: DatasetModel, hparams: dict, monkeypatch):
        """
        Should mark task as done even if training fails
        """
        task = TrainTaskModel(
            hparams=json.dumps(hparams), user=dataset.task.user, dataset=dataset
        )
        db.session.add(task)
        db.session.commit()

        word2vec_wrapper = MagicMock()
        word2vec_wrapper.train.side_effect = RuntimeError("error")
        monkeypatch.setattr(trainer, "word2vec_wrapper", word2vec_wrapper)

        with pytest.raises(RuntimeError):
            trainer.tick(db.session)

        assert task.model is None
        assert task.end_time is not None
