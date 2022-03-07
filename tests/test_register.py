from http import HTTPStatus

import pytest
from flask.testing import FlaskClient

from api.database import RegisterModel, db


@pytest.fixture()
def user_request(client: FlaskClient) -> RegisterModel:
    with client.application.app_context():
        model = RegisterModel(email="example@example.com", username="example")
        db.session.add(model)
        db.session.commit()
        yield model


class TestRegister:
    def test_request_account(self, client: FlaskClient):
        response = client.post(
            "/register", json={"email": "example@example.com", "username": "example"}
        )
        assert response.status_code == HTTPStatus.OK

    def test_request_duplicate_account(
        self, client: FlaskClient, user_request: RegisterModel
    ):
        response = client.post(
            "/register", json={"email": "example@example.com", "username": "example"}
        )
        assert response.status_code == HTTPStatus.CONFLICT

    def test_accept_request(self, client: FlaskClient, user_request: RegisterModel):
        response = client.get(f"/register/accept?accept=True&id={user_request.id}")
        assert response.status_code == HTTPStatus.OK

    def test_accept_invalid(self, client: FlaskClient):
        response = client.get("/register/accept?accept=True&id=99999")
        assert response.status_code == HTTPStatus.NOT_FOUND

    def test_decline_request(self, client: FlaskClient, user_request: RegisterModel):
        response = client.get(f"/register/accept?accept=False&id={user_request.id}")
        assert response.status_code == HTTPStatus.OK
