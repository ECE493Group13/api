import re
from http import HTTPStatus
from typing import Any

from flask.views import MethodView
from flask_smorest import Blueprint, abort
from marshmallow import Schema, fields, post_dump
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema

from api.authentication import auth
from api.database import (
    AnalogyTestResultModel,
    AnalogyTestResultRowModel,
    AnalogyTestTaskModel,
    TrainedModel,
    TrainTaskModel,
    db,
)

blueprint = Blueprint(
    "analogy-test-task", "analogy-test-task", url_prefix="/analogy-test-task"
)


class AnalogyTestPostSchema(Schema):
    trained_model_id = fields.Int(required=True)
    domain1_name = fields.Str(required=True)
    domain2_name = fields.Str(required=True)
    domain3_name = fields.Str(required=True)
    domain1_words = fields.List(fields.Str(), required=True)
    domain2_words = fields.List(fields.Str(), required=True)
    domain3_words = fields.List(fields.Str(), required=True)


class AnalogyTestListSchema(Schema):
    is_complete = fields.Bool()
    is_error = fields.Bool()


class AnalogyTestTaskSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = AnalogyTestTaskModel
        include_fk = True

    is_complete = fields.Bool()
    is_error = fields.Bool()

    # Convert words to lists when serializing
    domain1_words = fields.List(fields.Str(), required=True)
    domain2_words = fields.List(fields.Str(), required=True)
    domain3_words = fields.List(fields.Str(), required=True)

    @post_dump()
    def split_domain_words(self, data, **_kwargs):
        def unpack_words(words):
            words = "".join(words).split()
            words = [word.replace("_", " ") for word in words]
            return words

        return {
            **data,
            "domain1_words": unpack_words(data["domain1_words"]),
            "domain2_words": unpack_words(data["domain2_words"]),
            "domain3_words": unpack_words(data["domain3_words"]),
        }


@blueprint.route("")
class AnalogyTestTask(MethodView):
    @blueprint.arguments(AnalogyTestPostSchema, location="json")
    @blueprint.response(HTTPStatus.CREATED, AnalogyTestTaskSchema)
    def post(self, args: dict[str, Any]):

        trained_model = (
            db.session.query(TrainedModel)
            .join(TrainTaskModel)
            .filter(TrainTaskModel.user_id == auth.user.id)
            .filter(TrainedModel.id == args["trained_model_id"])
        ).one()

        def process_words(words: list[str]):
            words = [word.lower() for word in words]
            words = [re.sub(r"\s+", "_", word) for word in words]
            return " ".join(words)

        task = AnalogyTestTaskModel(
            user=auth.user,
            model=trained_model,
            domain1_name=args["domain1_name"],
            domain2_name=args["domain2_name"],
            domain3_name=args["domain3_name"],
            domain1_words=process_words(args["domain1_words"]),
            domain2_words=process_words(args["domain2_words"]),
            domain3_words=process_words(args["domain3_words"]),
        )
        db.session.add(task)
        db.session.commit()
        return task

    @blueprint.arguments(AnalogyTestListSchema, location="query")
    @blueprint.response(HTTPStatus.OK, AnalogyTestTaskSchema(many=True))
    def get(self, args: dict[str, Any]):
        is_complete = args.get("is_complete")
        is_error = args.get("is_error")

        query = db.session.query(AnalogyTestTaskModel).filter_by(user_id=auth.user.id)

        if is_complete is not None:
            query = query.filter(AnalogyTestTaskModel.is_complete.is_(is_complete))
        if is_error is not None:
            query = query.filter(AnalogyTestTaskModel.is_error.is_(is_error))

        return query.all()


@blueprint.route("/<int:analogy_test_task_id>")
class AnalogyTestTaskById(MethodView):
    @blueprint.response(HTTPStatus.OK, AnalogyTestTaskSchema)
    @blueprint.alt_response(HTTPStatus.NOT_FOUND)
    def get(self, analogy_test_task_id: int):
        task = (
            db.session.query(AnalogyTestTaskModel)
            .filter_by(user_id=auth.user.id)
            .filter_by(id=analogy_test_task_id)
            .one_or_none()
        )
        if task is None:
            abort(HTTPStatus.NOT_FOUND)

        return task


class AnalogyTestResultRowSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = AnalogyTestResultRowModel


class AnalogyTestResultSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = AnalogyTestResultModel

    rows = fields.Nested(AnalogyTestResultRowSchema, many=True)


@blueprint.route("/<int:analogy_test_task_id>/result")
class AnalogyTestResult(MethodView):
    @blueprint.response(HTTPStatus.OK, AnalogyTestResultSchema)
    @blueprint.alt_response(HTTPStatus.NOT_FOUND)
    def get(self, analogy_test_task_id: int):
        task: AnalogyTestTaskModel | None = (
            db.session.query(AnalogyTestTaskModel)
            .filter(AnalogyTestTaskModel.user_id == auth.user.id)
            .filter(AnalogyTestTaskModel.id == analogy_test_task_id)
        ).one_or_none()

        if task is None or task.result is None:
            abort(HTTPStatus.NOT_FOUND)

        return task.result
