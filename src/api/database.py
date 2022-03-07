from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, Text
from sqlalchemy.orm import relationship

db = SQLAlchemy()


class KeywordsModel(db.Model):
    __table_args__ = {"schema": "docs"}
    __tablename__ = "doc_keywords_0"

    dkey = Column(Text, primary_key=True)
    keywords = Column(Text)
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
