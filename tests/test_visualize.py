import json
from http import HTTPStatus

import pytest
from flask.testing import FlaskClient

from api.database import DatasetModel, TrainedModel, TrainTaskModel, UserModel, db


@pytest.fixture()
def visualize(authorized_user: UserModel):
    dataset = DatasetModel(num_papers=0, name="")

    task = TrainTaskModel(hparams=json.dumps({}), user=authorized_user, dataset=dataset)

    visualization = {"labels": ["hello", "world"], "x": [1, 2], "y": [3, 4]}
    trained_model = TrainedModel(
        embeddings_filename="embeddings.txt",
        visualization=json.dumps(visualization),
        task=task,
    )
    db.session.add_all([dataset, task, trained_model])
    db.session.commit()
    return trained_model


class TestVisualize:
    def test_get_success(
        self,
        client: FlaskClient,
        visualize: TrainedModel,
        auth_headers: dict,
    ):
        """
        Get Request to /visualize gets lables, x, y data for visualization
        """
        response = client.get(
            f"/visualize?train_task_id={visualize.task.id}", headers=auth_headers
        )
        assert response.status_code == HTTPStatus.OK
        assert response.json == json.dumps(
            {"labels": ["hello", "world"], "x": [1, 2], "y": [3, 4]}
        )

    def test_get_fail(self, client: FlaskClient, auth_headers: dict):
        """
        Get Request to /visualize for non existent id
        """
        response = client.get("/visualize?train_task_id=-1", headers=auth_headers)
        assert response.status_code == HTTPStatus.NOT_FOUND
