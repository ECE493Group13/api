import argon2
import pytest
from flask.testing import FlaskClient

from api import app, mail
from api.authentication import auth
from api.database import UserModel, db


@pytest.fixture()
def client():
    app.config.update({"TESTING": True})
    app.config.update({"SQLALCHEMY_DATABASE_URI": "sqlite://"})

    app.config.update({"MAIL_SUPPRESS_SEND": True})
    mail.init_app(app)
    test_client = app.test_client()

    with test_client.application.app_context():
        db.engine.execute("attach ':memory:' as docs")
        db.Model.metadata.create_all(db.engine)

    yield test_client

    with test_client.application.app_context():
        db.Model.metadata.drop_all(db.engine)
        db.engine.execute("detach docs")


ph = argon2.PasswordHasher()


@pytest.fixture()
def user(client: FlaskClient):
    with client.application.app_context():
        user_ = UserModel(
            username="example",
            password=ph.hash("password"),
            email="user@example.com",
            is_temp_password=False,
        )
        db.session.add(user_)
        db.session.commit()
        yield user_


@pytest.fixture()
def authorized_user(user: UserModel):
    auth.add_session(user)
    return user


@pytest.fixture()
def auth_headers(authorized_user: UserModel):
    return {"Authorization": f"Bearer {authorized_user.session.token}"}
