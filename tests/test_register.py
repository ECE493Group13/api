from http import HTTPStatus

import pytest
from flask.testing import FlaskClient

from api.database import RegisterModel, db


@pytest.fixture()
def user_request(client: FlaskClient) -> RegisterModel:
    model = RegisterModel(
        username="decline@example.com"
    )
    with client.application.app_context():
        db.session.add(model)
        db.session.commit()
    yield model


class TestRegister:
    id = None

    def test_request_account(self, client: FlaskClient):
        response = client.post(
            "/request", json={"username": "example@example.com"}
        )
        assert response.status_code == HTTPStatus.OK

    def test_request_duplicate_account(self, client: FlaskClient):
        response = client.post(
            "/request", json={"username": "example@example.com"}
        )
        assert response.status_code == HTTPStatus.CONFLICT

    def test_accept_request(self, client: FlaskClient, user_request: RegisterModel):
        response = client.get(
            f"/request/accept?accept=True&id={user_request.id}"
        )
        assert response.status_code == HTTPStatus.OK

    def test_accept_invalid(self, client: FlaskClient):
        response = client.get(
            "/request/accept?accept=True&id=99999"
        )
        assert response.status_code == HTTPStatus.NOT_FOUND

    def test_decline_request(self, client: FlaskClient, user_request: RegisterModel):
        response = client.get(
            f"/request/accept?accept=False&id={user_request.id}"
        )
        assert response.status_code == HTTPStatus.OK
