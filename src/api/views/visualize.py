from http import HTTPStatus

from flask.views import MethodView
from flask_smorest import Blueprint, abort
from marshmallow import Schema, fields

from api.authentication import auth
from api.database import TrainedModel, TrainTaskModel, db

blueprint = Blueprint("visualize", "visualize", url_prefix="/visualize")


class VisualizeSchema(Schema):
    train_task_id = fields.Int(required=True)


@blueprint.route("")
class VisualizeTask(MethodView):
    @blueprint.arguments(VisualizeSchema, location="query")
    @blueprint.response(HTTPStatus.OK)
    @blueprint.alt_response(HTTPStatus.NOT_FOUND)
    def get(self, args: dict[int]):
        train_task_id: int = args.get("train_task_id")

        train_task_model = (
            db.session.query(TrainTaskModel)
            .join(TrainedModel)
            .filter(TrainTaskModel.id == train_task_id)
            .filter(TrainTaskModel.user_id == auth.user.id)
        ).one_or_none()

        if train_task_model is None:
            abort(HTTPStatus.NOT_FOUND)

        model = train_task_model.model
        visualization = model.visualization

        return visualization
