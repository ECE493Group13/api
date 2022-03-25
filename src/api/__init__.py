from flask import Flask
from flask_cors import CORS
from flask_smorest import Api

from api.authentication import auth
from api.config import DatabaseConfig, MailConfig, OpenAPIConfig
from api.database import db
from api.mail import mail
from api.views.auth import blueprint as login_blueprint
from api.views.filter_task import blueprint as filter_task_blueprint
from api.views.healthcheck import blueprint as healthcheck_blueprint
from api.views.register import blueprint as register_blueprint
from api.views.train_task import blueprint as train_task_blueprint
from api.views.visualize import blueprint as visualize_blueprint

app = Flask(__name__)
app.config["API_TITLE"] = "DMS API"
app.config["API_VERSION"] = "0.0.1"
app.config.from_object(OpenAPIConfig)
app.config.from_object(DatabaseConfig)
app.config.from_object(MailConfig)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["AUTH_EXCLUDE_ENDPOINTS"] = [
    "healthcheck.Healthcheck",
    "auth.Login",
    "api-docs.openapi_rapidoc",
    "api-docs.openapi_swagger_ui",
    "api-docs.openapi_redoc",
    "api-docs.openapi_json",
    "register.Register",
    "register.AcceptRegister",
]

db.init_app(app)
auth.init_app(app)
mail.init_app(app)

CORS(app)

api = Api(app)

api.register_blueprint(filter_task_blueprint)
api.register_blueprint(healthcheck_blueprint)
api.register_blueprint(login_blueprint)
api.register_blueprint(register_blueprint)
api.register_blueprint(visualize_blueprint)
api.register_blueprint(train_task_blueprint)
