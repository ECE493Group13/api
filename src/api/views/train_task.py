import json
from http import HTTPStatus
from typing import Any

from flask.views import MethodView
from flask_smorest import Blueprint, abort
from marshmallow import Schema, fields
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema

from api.authentication import auth
from api.database import DatasetModel, FilterTaskModel, TrainTaskModel, db

blueprint = Blueprint("train-task", "train-task", url_prefix="/train-task")


class HyperparameterSchema(Schema):
    embedding_size = fields.Int(required=True)
    epochs_to_train = fields.Int(required=True)
    learning_rate = fields.Float(required=True)
    num_neg_samples = fields.Int(required=True)
    batch_size = fields.Int(required=True)
    concurrent_steps = fields.Int(required=True)
    window_size = fields.Int(required=True)
    min_count = fields.Int(required=True)
    subsample = fields.Float(required=True)


class TrainPostSchema(Schema):
    hparams = fields.Nested(HyperparameterSchema, required=True)
    dataset_id = fields.Int(required=True)


class TrainListSchema(Schema):
    dataset_id = fields.Int()


class TrainTaskSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = TrainTaskModel
        include_fk = True

    is_complete = fields.Bool()
    is_error = fields.Bool()


@blueprint.route("")
class TrainTask(MethodView):
    @blueprint.arguments(TrainPostSchema, location="json")
    @blueprint.response(HTTPStatus.CREATED, TrainTaskSchema)
    def post(self, args: dict[str, Any]):
        dataset_id: int = args["dataset_id"]
        hparams: dict = args["hparams"]

        dataset = (
            db.session.query(DatasetModel)
            .join(FilterTaskModel)
            .filter(FilterTaskModel.user_id == auth.user.id)
            .filter(DatasetModel.id == dataset_id)
        ).one()

        task = TrainTaskModel(
            hparams=json.dumps(hparams, sort_keys=True),
            user=auth.user,
            dataset=dataset,
        )
        db.session.add(task)
        db.session.commit()
        return task

    @blueprint.arguments(TrainListSchema, location="query")
    @blueprint.response(HTTPStatus.OK, TrainTaskSchema(many=True))
    def get(self, args: dict[str, Any]):
        dataset_id: int | None = args.get("dataset_id")

        query = db.session.query(TrainTaskModel).filter_by(user_id=auth.user.id)

        if dataset_id is not None:
            query = query.filter(TrainTaskModel.dataset_id == dataset_id)

        return query.all()


@blueprint.route("/<int:train_task_id>")
class TrainTaskById(MethodView):
    @blueprint.response(HTTPStatus.OK, TrainTaskSchema)
    @blueprint.alt_response(HTTPStatus.NOT_FOUND)
    def get(self, train_task_id: int):
        train_task = (
            db.session.query(TrainTaskModel)
            .filter_by(user_id=auth.user.id)
            .filter_by(id=train_task_id)
            .one_or_none()
        )
        if train_task is None:
            abort(HTTPStatus.NOT_FOUND)

        return train_task


@blueprint.route("/suggest-hparams")
class SuggestHParams(MethodView):
    @blueprint.response(HTTPStatus.OK, HyperparameterSchema)
    def get(self):
        return {
            "embedding_size": 200,
            "epochs_to_train": 15,
            "learning_rate": 0.025,
            "num_neg_samples": 25,
            "batch_size": 500,
            "concurrent_steps": 12,
            "window_size": 5,
            "min_count": 5,
            "subsample": 1e-3,
        }
