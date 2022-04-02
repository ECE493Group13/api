from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    Text,
)
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship

db = SQLAlchemy()


class PaperModel(db.Model):
    """
    ORM class for the paper info table from the general index.

    This real table has no primary keys, but primary keys have been added here
    to stop SQLAlchemy from complaining.
    """

    __table_args__ = {"schema": "docs"}
    __tablename__ = "doc_meta_0"

    dkey = Column(Text, primary_key=True)
    meta_doi = Column(Text)
    doi = Column(Text)
    doc_doi = Column(Text)
    doc_pub_date = Column(DateTime)
    pub_date = Column(DateTime)
    meta_pub_date = Column(DateTime)
    doc_author = Column(Text)
    meta_author = Column(Text)
    author = Column(Text)
    doc_title = Column(Text)
    meta_title = Column(Text)
    title = Column(Text)


class NgramModel(db.Model):
    """
    ORM class for the ngram table from the general index.

    This real table has no primary keys, but primary keys have been added here
    to stop SQLAlchemy from complaining.
    """

    __table_args__ = {"schema": "docs"}
    __tablename__ = "doc_ngrams_0"

    ngram = Column(Text, primary_key=True)
    ngram_lc = Column(Text)
    ngram_tokens = Column(Integer)
    ngram_count = Column(Integer)
    term_freq = Column(Float)
    doc_count = Column(Integer)
    insert_date = Column(DateTime)

    dkey = Column(Text, ForeignKey("docs.doc_meta_0.dkey"), primary_key=True)


class KeywordsModel(db.Model):
    """
    ORM class for the keywords table from the general index.

    This real table has no primary keys, but primary keys have been added here
    to stop SQLAlchemy from complaining.
    """

    __table_args__ = {"schema": "docs"}
    __tablename__ = "doc_keywords_0"

    dkey = Column(Text, primary_key=True)
    keywords = Column(Text, primary_key=True)
    keywords_lc = Column(Text)
    keyword_tokens = Column(Integer)
    keyword_score = Column(Float)
    doc_count = Column(Integer)
    insert_date = Column(DateTime)


class UserModel(db.Model):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    email = Column(Text, nullable=False, unique=True)
    username = Column(Text, nullable=False, unique=True)
    password = Column(Text, nullable=False)
    is_temp_password = Column(Boolean, nullable=False)

    session = relationship("SessionModel", uselist=False, back_populates="user")


class SessionModel(db.Model):
    __tablename__ = "session"

    token = Column(Text, primary_key=True)
    created = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_refresh = Column(DateTime, nullable=False, default=datetime.utcnow)

    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    user = relationship("UserModel", uselist=False, back_populates="session")


class RegisterModel(db.Model):
    __tablename__ = "register"

    id = Column(Integer, primary_key=True)
    email = Column(Text, unique=True)
    username = Column(Text, unique=True)
    accepted = Column(Boolean, default=False)
    accept_key = Column(Text)


class FilterTaskModel(db.Model):
    __tablename__ = "filter_task"

    id = Column(Integer, primary_key=True)
    created = Column(DateTime, nullable=False, default=datetime.utcnow)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    keywords = Column(Text, nullable=False)

    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    user = relationship("UserModel", uselist=False)

    dataset_id = Column(Integer, ForeignKey("dataset.id"), nullable=True)
    dataset = relationship("DatasetModel", uselist=False, back_populates="task")

    @hybrid_property
    def is_complete(self):
        return self.end_time is not None

    @is_complete.expression
    def is_complete(cls):  # pylint: disable=no-self-argument
        return cls.end_time.isnot(None)

    @hybrid_property
    def is_error(self):
        return self.dataset_id is None

    @is_error.expression
    def is_error(cls):  # pylint: disable=no-self-argument
        return cls.dataset_id.is_(None) & cls.is_complete

    # Hacks to make pylint work
    is_complete: Column
    is_error: Column


class DatasetModel(db.Model):
    __tablename__ = "dataset"

    id = Column(Integer, primary_key=True)

    task = relationship("FilterTaskModel", uselist=False, back_populates="dataset")


class DatasetPaperModel(db.Model):
    """Association table for dataset-paper relation"""

    __tablename__ = "dataset_paper"

    dataset_id = Column(
        Integer, ForeignKey("dataset.id"), nullable=False, primary_key=True
    )
    dkey = Column(
        Text, ForeignKey("docs.doc_meta_0.dkey"), nullable=False, primary_key=True
    )


class TrainTaskModel(db.Model):
    __tablename__ = "train_task"

    id = Column(Integer, primary_key=True)
    hparams = Column(Text, nullable=False)
    created = Column(DateTime, nullable=False, default=datetime.utcnow)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)

    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    user = relationship("UserModel", uselist=False)

    dataset_id = Column(Integer, ForeignKey("dataset.id"), nullable=False)
    dataset = relationship("DatasetModel", uselist=False)

    model_id = Column(Integer, ForeignKey("trained_model.id"), nullable=True)
    model = relationship("TrainedModel", uselist=False, back_populates="task")


class TrainedModel(db.Model):
    __tablename__ = "trained_model"

    id = Column(Integer, primary_key=True)
    data = Column(LargeBinary, nullable=False)

    task = relationship("TrainTaskModel", uselist=False, back_populates="model")
