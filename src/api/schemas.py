from marshmallow_sqlalchemy import SQLAlchemyAutoSchema

from api.database import DatasetModel


class DatasetSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = DatasetModel
        include_fk = True
