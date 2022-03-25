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
    is_temp_password = fields.Bool()


class UpdatePasswordSchema(Schema):
    old_password = fields.Str(required=True)
    new_password = fields.Str(required=True)


class UpdatePasswordResponseSchema(Schema):
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
        is_temp_password = user.is_temp_password
        return {"token": token, "is_temp_password": is_temp_password}


@blueprint.route("/logout")
class Logout(MethodView):
    @blueprint.response(HTTPStatus.NO_CONTENT)
    def post(self):
        auth.remove_session()


@blueprint.route("/update-password")
class UpdatePassword(MethodView):
    @blueprint.arguments(UpdatePasswordSchema, location="json")
    @blueprint.response(HTTPStatus.OK, UpdatePasswordResponseSchema)
    @blueprint.alt_response(HTTPStatus.UNAUTHORIZED)
    def post(self, args: dict[str, str]):
        old_password = args["old_password"]
        new_password = args["new_password"]

        try:
            ph.verify(auth.user.password, old_password)
        except argon2.exceptions.VerificationError:
            logger.exception("Password mismatch")
            abort(HTTPStatus.UNAUTHORIZED)

        auth.user.password = ph.hash(new_password)
        auth.user.is_temp_password = False
        db.session.commit()
        token = auth.refresh_session().token
        return {"token": token}


@blueprint.route("/ping")
class Ping(MethodView):
    @blueprint.response(HTTPStatus.NO_CONTENT)
    def post(self):
        pass
