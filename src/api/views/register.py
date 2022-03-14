import secrets
import string
from http import HTTPStatus

import argon2
from flask import request
from flask.views import MethodView
from flask_mail import Message
from flask_smorest import Blueprint, abort
from marshmallow import Schema, fields

from api.config import MailConfig
from api.database import RegisterModel, UserModel, db
from api.mail import mail

blueprint = Blueprint("register", "register", url_prefix="/register")

ph = argon2.PasswordHasher()


class AcceptRegisterSchema(Schema):
    accept_key = fields.Str(required=True)
    accept = fields.Boolean(required=True)
    id = fields.Integer(required=True)


class RegisterSchema(Schema):
    email = fields.Str(required=True)
    username = fields.Str(required=True)


@blueprint.route("")
class Register(MethodView):
    @blueprint.arguments(RegisterSchema, location="json")
    @blueprint.response(HTTPStatus.NO_CONTENT)
    @blueprint.alt_response(HTTPStatus.CONFLICT)
    def post(self, args: dict[str, str]):
        email = args["email"]
        username = args["username"]

        register_user: RegisterModel = (
            db.session.query(RegisterModel)
            .filter(
                (RegisterModel.email == email) | (RegisterModel.username == username)
            )
            .one_or_none()
        )

        # User already requested an account
        if register_user is not None:
            abort(HTTPStatus.CONFLICT)

        user: UserModel = (
            db.session.query(UserModel)
            .filter((UserModel.email == email) | (UserModel.username == username))
            .one_or_none()
        )

        # User already has an account
        if user is not None:
            abort(HTTPStatus.CONFLICT)

        alphabet = string.ascii_letters + string.digits
        accept_key = "".join(secrets.choice(alphabet) for _ in range(20))

        register_user = RegisterModel(
            email=email, username=username, accepted=False, accept_key=accept_key
        )

        db.session.add(register_user)
        db.session.commit()

        register_user: RegisterModel = (
            db.session.query(RegisterModel).filter_by(email=email).one_or_none()
        )

        html = f"""{email} is requesting an account: \
            <form method="POST" action="{request.base_url}/accept">
                <input type="hidden" id="accept" name="accept" value=True>
                <input type="hidden" id="id" name="id" value={register_user.id}>
                <input type="hidden" id="accept_key" name="accept_key" value={accept_key}>
                <button type="submit">Accept</button>
            </form>
            <form method="POST" action="{request.base_url}/accept">
                <input type="hidden" id="accept" name="accept" value=False>
                <input type="hidden" id="id" name="id" value={register_user.id}>
                <input type="hidden" id="accept_key" name="accept_key" value={accept_key}>
                <button type="submit">Decline</button>
            </form>
            """
        msg = Message(
            "Account Request for DMS",
            sender=MailConfig.MAIL_USERNAME,
            recipients=[MailConfig.MAIL_USERNAME],
            html=html,
        )
        mail.send(msg)
        return HTTPStatus.NO_CONTENT


@blueprint.route("/accept")
class AcceptRegister(MethodView):
    @blueprint.arguments(AcceptRegisterSchema, location="form")
    @blueprint.response(HTTPStatus.NO_CONTENT)
    @blueprint.alt_response(HTTPStatus.NOT_FOUND)
    @blueprint.alt_response(HTTPStatus.ALREADY_REPORTED)
    @blueprint.alt_response(HTTPStatus.UNAUTHORIZED)
    def post(self, args: dict[bool, str, int]):
        accept = args["accept"]
        accept_key = args["accept_key"]
        register_id = args["id"]

        register_user: RegisterModel = (
            db.session.query(RegisterModel).filter_by(id=register_id).one_or_none()
        )

        # Request not found
        if register_user is None:
            abort(HTTPStatus.NOT_FOUND)

        if register_user.accept_key != accept_key:
            abort(HTTPStatus.UNAUTHORIZED)

        if accept:
            alphabet = string.ascii_letters + string.digits
            password = "".join(secrets.choice(alphabet) for _ in range(20))
            password_hash = ph.hash(password)
            created_user = UserModel(
                email=register_user.email,
                username=register_user.username,
                password=password_hash,
                is_temp_password=True,
            )
            db.session.add(created_user)

        db.session.delete(register_user)
        db.session.commit()

        body = f"Your account request has been {'granted' if accept else 'denied'}.\n"
        if accept:
            body += f"Username: {register_user.username}\nPassword: {password}"

        msg = Message(
            f"Account Request for DMS {'Approved' if accept else 'Denied'}",
            sender=MailConfig.MAIL_USERNAME,
            recipients=[register_user.email],
            body=body,
        )
        mail.send(msg)
        return HTTPStatus.NO_CONTENT
