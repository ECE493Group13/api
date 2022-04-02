from http import HTTPStatus
from typing import Any

from flask.views import MethodView
from flask_smorest import Blueprint
from gensim.models import KeyedVectors
from marshmallow import Schema, fields

from api.authentication import auth
from api.database import TrainedModel, TrainTaskModel, db

blueprint = Blueprint("verify", "verify", url_prefix="/verify")


class WordProximitySchema(Schema):
    word = fields.Str()
    proximity = fields.Float()


class MostSimilarSchema(Schema):
    trained_model_id = fields.Int(required=True)
    word = fields.Str(required=True)
    count = fields.Int(load_default=100)


class AnalogyTestSchema(Schema):
    trained_model_id = fields.Int(required=True)
    word_a = fields.Str(required=True)
    word_b = fields.Str(required=True)
    word_c = fields.Str(required=True)
    count = fields.Int(load_default=500)


def keyed_vectors_from_model(model: TrainedModel) -> KeyedVectors:
    return KeyedVectors.load_word2vec_format(model.embeddings_filename)


@blueprint.route("/most-similar")
class MostSimilar(MethodView):
    @blueprint.arguments(MostSimilarSchema, location="query")
    @blueprint.response(HTTPStatus.OK, WordProximitySchema(many=True))
    def get(self, args: dict[str, Any]):
        model_id: int = args["trained_model_id"]
        word: str = args["word"]
        count: int = args["count"]

        model = (
            db.session.query(TrainedModel)
            .join(TrainTaskModel)
            .filter(TrainTaskModel.user_id == auth.user.id)
            .filter(TrainedModel.id == model_id)
            .one()
        )

        keyed_vectors = keyed_vectors_from_model(model)
        try:
            return [
                {"word": word_, "proximity": proximity}
                for word_, proximity in keyed_vectors.most_similar(word, topn=count)
            ]
        except KeyError:
            return []


@blueprint.route("/analogy-test")
class AnalogyTest(MethodView):
    @blueprint.arguments(AnalogyTestSchema, location="query")
    @blueprint.response(HTTPStatus.OK, WordProximitySchema(many=True))
    def get(self, args: dict[str, Any]):
        model_id: int = args["trained_model_id"]
        word_a: str = args["word_a"]
        word_b: str = args["word_b"]
        word_c: str = args["word_c"]
        count: int = args["count"]

        model = (
            db.session.query(TrainedModel)
            .join(TrainTaskModel)
            .filter(TrainTaskModel.user_id == auth.user.id)
            .filter(TrainedModel.id == model_id)
            .one()
        )

        keyed_vectors = keyed_vectors_from_model(model)
        try:
            vector_d = (
                keyed_vectors[word_c] + keyed_vectors[word_b] - keyed_vectors[word_a]
            )
            return [
                {"word": word, "proximity": proximity}
                for word, proximity in keyed_vectors.similar_by_vector(
                    vector_d, topn=count
                )
            ]
        except KeyError:
            return []
