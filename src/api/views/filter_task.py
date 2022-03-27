from http import HTTPStatus
from typing import Any

from flask.views import MethodView
from flask_smorest import Blueprint, abort
from marshmallow import Schema, fields
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema

from api.authentication import auth
from api.database import FilterTaskModel, db

blueprint = Blueprint("filter-task", "filter-task", url_prefix="/filter-task")


class FilterPostSchema(Schema):
    keywords = fields.List(fields.Str(), required=True)


class FilterListSchema(Schema):
    is_complete = fields.Bool()
    is_error = fields.Bool()


class FilterTaskSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = FilterTaskModel
        include_fk = True

    is_complete = fields.Bool()
    is_error = fields.Bool()


@blueprint.route("")
class FilterTask(MethodView):
    @blueprint.arguments(FilterPostSchema, location="json")
    @blueprint.response(HTTPStatus.CREATED, FilterTaskSchema)
    def post(self, args: dict[str, Any]):
        keywords: list[str] = args["keywords"]

        task = FilterTaskModel(keywords=" ".join(keywords), user=auth.user)
        db.session.add(task)
        db.session.commit()
        return task

    @blueprint.arguments(FilterListSchema, location="query")
    @blueprint.response(HTTPStatus.OK, FilterTaskSchema(many=True))
    def get(self, args: dict[str, Any]):
        is_complete = args.get("is_complete")
        is_error = args.get("is_error")

        query = db.session.query(FilterTaskModel).filter_by(user_id=auth.user.id)

        if is_complete is not None:
            query = query.filter(FilterTaskModel.is_complete.is_(is_complete))
        if is_error is not None:
            query = query.filter(FilterTaskModel.is_error.is_(is_error))

        return query.all()


@blueprint.route("/<int:filter_task_id>")
class FilterTaskById(MethodView):
    @blueprint.response(HTTPStatus.OK, FilterTaskSchema)
    @blueprint.alt_response(HTTPStatus.NOT_FOUND)
    def get(self, filter_task_id: int):
        filter_task = (
            db.session.query(FilterTaskModel)
            .filter_by(user_id=auth.user.id)
            .filter_by(id=filter_task_id)
            .one_or_none()
        )
        if filter_task is None:
            abort(HTTPStatus.NOT_FOUND)

        return filter_task
