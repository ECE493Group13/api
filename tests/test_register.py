from http import HTTPStatus

import pytest
from flask.testing import FlaskClient

from api.database import RegisterModel, db


@pytest.fixture()
def user_request(client: FlaskClient) -> RegisterModel:
    with client.application.app_context():
        model = RegisterModel(
            email="example@example.com", username="example", accept_key="test"
        )
        db.session.add(model)
        db.session.commit()
        yield model


class TestRegister:
    def test_request_account(self, client: FlaskClient):
        response = client.post(
            "/register", json={"email": "example@example.com", "username": "example"}
        )
        assert response.status_code == HTTPStatus.NO_CONTENT

    def test_request_duplicate_account(
        self, client: FlaskClient, user_request: RegisterModel
    ):
        response = client.post(
            "/register", json={"email": "example@example.com", "username": "example"}
        )
        assert response.status_code == HTTPStatus.CONFLICT

    def test_accept_request(self, client: FlaskClient, user_request: RegisterModel):
        response = client.post(
            "/register/accept",
            data={"accept": True, "id": user_request.id, "accept_key": "test"},
        )
        assert response.status_code == HTTPStatus.NO_CONTENT

    def test_accept_invalid(self, client: FlaskClient, user_request: RegisterModel):
        response = client.post(
            "/register/accept",
            data={"accept": True, "id": "9999", "accept_key": "test"},
        )
        assert response.status_code == HTTPStatus.NOT_FOUND

        response = client.post(
            "/register/accept",
            data={"accept": True, "id": user_request.id, "accept_key": "wrongkey"},
        )
        assert response.status_code == HTTPStatus.UNAUTHORIZED

    def test_decline_request(self, client: FlaskClient, user_request: RegisterModel):
        response = client.post(
            "/register/accept",
            data={"accept": False, "id": user_request.id, "accept_key": "test"},
        )
        assert response.status_code == HTTPStatus.NO_CONTENT
