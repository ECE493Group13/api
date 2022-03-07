import os


# https://flask-smorest.readthedocs.io/en/latest/openapi.html#serve-the-openapi-documentation
class OpenAPIConfig:
    OPENAPI_VERSION = "3.0.2"
    OPENAPI_JSON_PATH = "api-spec.json"
    OPENAPI_URL_PREFIX = "/"
    OPENAPI_REDOC_PATH = "/redoc"
    OPENAPI_REDOC_URL = (
        "https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js"
    )
    OPENAPI_SWAGGER_UI_PATH = "/swagger-ui"
    OPENAPI_SWAGGER_UI_URL = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"
    OPENAPI_RAPIDOC_PATH = "/rapidoc"
    OPENAPI_RAPIDOC_URL = "https://unpkg.com/rapidoc/dist/rapidoc-min.js"


class DatabaseConfig:
    USER = "postgres"
    HOST = "localhost"
    DATABASE = "postgres"
    PORT = "5433"
    PASSWORD = os.environ.get("DB_PASSWORD", "test")
    SQLALCHEMY_DATABASE_URI = (
        f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
    )


class MailConfig:
    MAIL_SERVER = "smtp.gmail.com"
    MAIL_PORT = 465
    MAIL_USERNAME = "dataminingsystem@gmail.com"
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "test")
    MAIL_USE_TLS = False
    MAIL_USE_SSL = True
    MAIL_DEBUG = True
    MAIL_SUPPRESS_SEND = False
