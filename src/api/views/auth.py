from http import HTTPStatus

import argon2
from flask.views import MethodView
from flask_smorest import Blueprint, abort
from logzero import logger
from marshmallow import Schema, fields

from api.authentication import auth
from api.database import UserModel, db

blueprint = Blueprint("auth", "auth", url_prefix="/auth")


class LoginSchema(Schema):
    username = fields.Str(required=True)
    password = fields.Str(required=True)


class LoginResponseSchema(Schema):
    token = fields.Str()


ph = argon2.PasswordHasher()


@blueprint.route("/login")
class Login(MethodView):
    @blueprint.arguments(LoginSchema, location="json")
    @blueprint.response(HTTPStatus.OK, LoginResponseSchema)
    @blueprint.alt_response(HTTPStatus.UNAUTHORIZED)
    def post(self, args: dict[str, str]):
        username = args["username"]
        password = args["password"]

        if auth.authenticated:
            auth.remove_session()

        user: UserModel = (
            db.session.query(UserModel).filter_by(username=username).one_or_none()
        )
        if user is None:
            abort(HTTPStatus.UNAUTHORIZED)

        try:
            ph.verify(user.password, password)
        except argon2.exceptions.VerificationError:
            logger.exception("Password mismatch")
            abort(HTTPStatus.UNAUTHORIZED)

        if ph.check_needs_rehash(user.password):
            user.password = ph.hash(password)

        token = auth.add_session(user).token
        return {"token": token}


@blueprint.route("/logout")
class Logout(MethodView):
    @blueprint.response(HTTPStatus.NO_CONTENT)
    def post(self):
        auth.remove_session()