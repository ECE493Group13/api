import secrets
import string

from http import HTTPStatus

from flask import request
from flask.views import MethodView
from flask_mail import Message
from flask_smorest import Blueprint, abort
from marshmallow import Schema, fields

from api.database import RegisterModel, UserModel, db
from api.mail import mail

from api.config import MailConfig

blueprint = Blueprint("register", "register", url_prefix="/register")


class AcceptRegisterSchema(Schema):
    accept = fields.Boolean(required=True)
    id = fields.Integer(required=True)


class RegisterSchema(Schema):
    username = fields.Str(required=True)


@blueprint.route("")
class Register(MethodView):
    @blueprint.arguments(RegisterSchema, location="json")
    @blueprint.response(HTTPStatus.OK)
    @blueprint.alt_response(HTTPStatus.CONFLICT)
    def post(self, args: dict[str, str]):
        username = args["username"]

        user: RegisterModel = (
            db.session.query(RegisterModel).filter_by(
                username=username).one_or_none()
        )

        # User already requested an account
        if user is not None:
            abort(HTTPStatus.CONFLICT)

        user = RegisterModel(username=username, accepted=False)

        db.session.add(user)
        db.session.commit()

        user: RegisterModel = (
            db.session.query(RegisterModel).filter_by(
                username=username).one_or_none()
        )

        html = f'{username} is requesting an account: <a href="{request.base_url}/accept?accept=True&id={user.id}">Accept</a> \
        <a href="{request.base_url}/accept?accept=False&id={user.id}">Reject</a>'
        msg = Message(
            "Account Request for DMS",
            sender=MailConfig.MAIL_USERNAME,
            recipients=[MailConfig.MAIL_USERNAME],
            html=html,
        )
        mail.send(msg)
        return HTTPStatus.OK


@blueprint.route("/accept")
class AcceptRegister(MethodView):
    @blueprint.arguments(AcceptRegisterSchema, location="query")
    @blueprint.response(HTTPStatus.OK)
    @blueprint.alt_response(HTTPStatus.NOT_FOUND)
    @blueprint.alt_response(HTTPStatus.ALREADY_REPORTED)
    def get(self, args: dict[int, bool]):
        accept = args["accept"]
        id = args["id"]

        user: RegisterModel = (
            db.session.query(RegisterModel).filter_by(id=id).one_or_none()
        )

        # Request not found
        if user is None:
            abort(HTTPStatus.NOT_FOUND)

        if accept:
            alphabet = string.ascii_letters + string.digits
            password = "".join(secrets.choice(alphabet) for _ in range(8))
            createUser = UserModel(
                email=user.username,
                username=user.username,
                password=password,
                is_temp_password=True,
            )
            db.session.add(createUser)

        db.session.delete(user)
        db.session.commit()

        body = f"Your account request has been {'granted' if accept else 'denied'}.\n"
        if accept:
            body += f"Username: {user.username}\nPassword: {password}"

        msg = Message(
            f"Account Request for DMS {'Approved' if accept else 'Denied'}",
            sender=MailConfig.MAIL_USERNAME,
            recipients=[user.username],
            body=body,
        )
        mail.send(msg)
        return HTTPStatus.OK
