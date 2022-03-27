from datetime import datetime
from http import HTTPStatus

import pytest
from flask.testing import FlaskClient

from api.database import (
    DatasetModel,
    DatasetPaperModel,
    FilterTaskModel,
    KeywordsModel,
    PaperModel,
    UserModel,
    db,
)
from api.workers.filterer import FilterWorker
from api.workers.worker import WorkerRunner


@pytest.fixture()
def keywords(papers: list[PaperModel]):
    paper1, paper2 = papers
    keywords1 = KeywordsModel(
        dkey=paper1.dkey,
        keywords="back pain",
        keywords_lc="back pain",
        keyword_tokens=2,
        keyword_score=1.0,
        doc_count=1,
    )
    keywords2 = KeywordsModel(
        dkey=paper2.dkey,
        keywords="pain",
        keywords_lc="pain",
        keyword_tokens=1,
        keyword_score=1.0,
        doc_count=1,
    )
    db.session.add_all([keywords1, keywords2])
    db.session.commit()
    return [keywords1, keywords2]


@pytest.fixture()
def filter_tasks(authorized_user: UserModel):
    dataset = DatasetModel()
    tasks = [
        FilterTaskModel(user=authorized_user, keywords="hello"),
        FilterTaskModel(
            user=authorized_user,
            keywords="hello world",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
        ),
        FilterTaskModel(
            user=authorized_user,
            keywords="world",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            dataset=dataset,
        ),
    ]
    db.session.add_all([*tasks, dataset])
    db.session.commit()
    return tasks


class TestFilterTask:
    def test_post(self, client: FlaskClient, auth_headers: dict, papers, keywords):
        response = client.post(
            "/filter-task", json={"keywords": ["back pain"]}, headers=auth_headers
        )
        assert response.status_code == HTTPStatus.CREATED
        assert "is_complete" in response.json
        assert "is_error" in response.json

        task: FilterTaskModel = (
            db.session.query(FilterTaskModel).filter_by(id=response.json["id"]).one()
        )
        WorkerRunner(FilterWorker())._tick(db.session)  # pylint: disable=W0212
        db.session.commit()

        assert task.keywords == "back pain"
        assert task.dataset is not None
        assert task.is_complete is True
        assert task.is_error is False

        papers = (
            db.session.query(PaperModel)
            .join(DatasetPaperModel)
            .join(DatasetModel)
            .filter(DatasetModel.id == task.dataset_id)
            .all()
        )

        assert len(papers) == 1
        assert papers[0].dkey == "doc1"

    def test_list(self, client: FlaskClient, auth_headers: dict, filter_tasks):
        """
        GET /filter-task should list filter tasks for the current user,
        and be filterable by complete and error status.
        """
        response = client.get("/filter-task", headers=auth_headers)
        assert response.status_code == HTTPStatus.OK
        assert len(response.json) == 3

        response = client.get("/filter-task?is_complete=true", headers=auth_headers)
        assert response.status_code == HTTPStatus.OK
        assert len(response.json) == 2

        response = client.get("/filter-task?is_error=true", headers=auth_headers)
        assert response.status_code == HTTPStatus.OK
        assert len(response.json) == 1

        response = client.get(
            "/filter-task?is_complete=true&is_error=true", headers=auth_headers
        )
        assert response.status_code == HTTPStatus.OK
        assert len(response.json) == 1
