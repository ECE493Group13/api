from http import HTTPStatus

import pytest
from flask.testing import FlaskClient

from api.database import (
    AnalogyTestTaskModel,
    TrainedModel,
    TrainTaskModel,
    UserModel,
    db,
)


@pytest.fixture()
def trained_model(authorized_user: UserModel):
    task = TrainTaskModel(user=authorized_user, hparams="", dataset_id=0)
    model = TrainedModel(task=task, data=b"2 1\nhello\t0.5\nworld\t0.5\n")
    db.session.add_all([task, model])
    db.session.commit()
    yield model


@pytest.fixture()
def analogy_test_task(trained_model: TrainedModel, authorized_user: UserModel):
    task = AnalogyTestTaskModel(
        user=authorized_user,
        model_id=trained_model.id,
        domain1_name="Anatomy",
        domain2_name="Pathology",
        domain3_name="Treatment",
        domain1_words="annulus cervical_spine",
        domain2_words="adjacent_disc_disease adolescent_ideopathic_scoliosis",
        domain3_words="abdominal acetaminophen",
    )
    db.session.add(task)
    db.session.commit()
    return task


class TestAnalogyTestTask:
    def test_post(
        self, client: FlaskClient, auth_headers: dict, trained_model: TrainedModel
    ):
        response = client.post(
            "/analogy-test-task",
            json={
                "trained_model_id": trained_model.id,
                "domain1_name": "Anatomy",
                "domain2_name": "Pathology",
                "domain3_name": "Treatment",
                "domain1_words": ["Annulus", "Cervical spine"],
                "domain2_words": [
                    "Adjacent disc disease",
                    "Adolescent ideopathic scoliosis",
                ],
                "domain3_words": ["abdominal", "acetaminophen"],
            },
            headers=auth_headers,
        )
        assert response.status_code == HTTPStatus.CREATED
        assert response.json["is_complete"] is False
        assert response.json["is_error"] is False
        assert response.json["domain1_name"] == "Anatomy"
        assert response.json["domain1_words"] == ["annulus", "cervical spine"]

    def test_list(self, client: FlaskClient, auth_headers: dict, analogy_test_task):
        response = client.get("/analogy-test-task", headers=auth_headers)
        assert response.status_code == HTTPStatus.OK
        assert len(response.json) == 1
        assert response.json[0]["domain1_name"] == "Anatomy"
        assert response.json[0]["domain1_words"] == ["annulus", "cervical spine"]

    def test_get(
        self,
        client: FlaskClient,
        auth_headers: dict,
        analogy_test_task: AnalogyTestTaskModel,
    ):
        id_ = analogy_test_task.id
        response = client.get(f"/analogy-test-task/{id_}", headers=auth_headers)
        assert response.status_code == HTTPStatus.OK
        assert response.json["domain1_name"] == "Anatomy"
        assert response.json["domain1_words"] == ["annulus", "cervical spine"]
