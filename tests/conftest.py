from datetime import datetime

import argon2
import pytest
from flask.testing import FlaskClient

from api import app, mail
from api.authentication import auth
from api.database import NgramModel, PaperModel, UserModel, db


@pytest.fixture()
def client():
    app.config.update({"TESTING": True})
    app.config.update({"SQLALCHEMY_DATABASE_URI": "sqlite://"})

    app.config.update({"MAIL_SUPPRESS_SEND": True})
    mail.init_app(app)
    test_client = app.test_client()

    with test_client.application.app_context():
        db.engine.execute("attach ':memory:' as docs")
        db.Model.metadata.create_all(db.engine)

    yield test_client

    with test_client.application.app_context():
        db.Model.metadata.drop_all(db.engine)
        db.engine.execute("detach docs")


ph = argon2.PasswordHasher()


@pytest.fixture()
def user(client: FlaskClient):
    with client.application.app_context():
        user_ = UserModel(
            username="example",
            password=ph.hash("password"),
            email="user@example.com",
            is_temp_password=False,
        )
        db.session.add(user_)
        db.session.commit()
        yield user_


@pytest.fixture()
def authorized_user(user: UserModel):
    auth.add_session(user)
    return user


@pytest.fixture()
def auth_headers(authorized_user: UserModel):
    return {"Authorization": f"Bearer {authorized_user.session.token}"}


TEXT = """
Aliqua incididunt officia magna laboris in qui excepteur adipisicing incididunt consectetur sunt et est. Non enim qui mollit enim sit eu nulla tempor deserunt aliqua dolore. Commodo sunt laborum ex Lorem do esse qui occaecat enim. Magna enim magna proident Lorem esse laboris labore ea enim proident quis ex deserunt. Deserunt proident consequat reprehenderit labore cupidatat officia tempor. Cillum quis irure laborum labore dolor est consectetur incididunt excepteur duis tempor aliquip magna sint.

Occaecat occaecat culpa laborum et in dolore elit ea ipsum mollit Lorem in veniam. Sint elit ea sunt Lorem ut aliqua minim officia mollit ipsum ea esse commodo. Magna culpa commodo veniam anim. Nulla magna aute non deserunt mollit. Fugiat voluptate voluptate velit amet et sint adipisicing enim mollit eu id. Non id labore do voluptate qui labore culpa ut eiusmod nulla reprehenderit voluptate eu ipsum.

Occaecat officia ea nulla consectetur ad. Laboris nostrud laborum sint incididunt cupidatat exercitation cupidatat Lorem quis ullamco anim aliqua in. Anim velit reprehenderit dolore occaecat anim. Incididunt sint laboris aute et laborum. Aute Lorem ex velit velit veniam occaecat exercitation. Aliqua irure ullamco adipisicing consectetur nulla occaecat dolore.

Lorem ex duis amet sunt exercitation. Dolore adipisicing Lorem non ullamco exercitation ut. Enim pariatur ullamco dolor aliqua est nulla consequat quis est anim aliquip.

Aute aliquip id est eu adipisicing velit est enim qui laborum. In duis commodo aliqua ad incididunt. Voluptate aliqua officia laboris pariatur eiusmod commodo ut laborum exercitation esse. Eiusmod enim ea dolore ut laborum minim id nulla mollit incididunt aliquip esse adipisicing. Reprehenderit ex consequat ad labore sint mollit sint id irure ad elit exercitation velit ipsum.

Lorem excepteur cupidatat tempor aliqua. Et ut officia laborum consectetur nulla fugiat do ea amet nisi. Incididunt incididunt incididunt aliquip amet. Sint esse sunt aliquip officia ea aliquip veniam velit eu anim laboris sint. Laborum laboris adipisicing nisi veniam nulla enim in in dolor culpa dolor. Consequat adipisicing duis sunt labore aute id Lorem. Consectetur laborum ad ea exercitation reprehenderit minim dolor enim pariatur in mollit proident.
"""


def make_ngrams(doc: PaperModel, text: str):
    tokens = text.split()
    ngrams = []
    for ngram_length in range(1, 6):
        for i in range(len(tokens)):
            ngram = tokens[i : i + ngram_length]
            if len(ngram) < ngram_length:
                continue
            ngram = " ".join(ngram)
            ngrams.append(ngram)

    ngram_counts = {}
    for ngram in ngrams:
        if ngram not in ngram_counts:
            ngram_counts[ngram] = 0
        ngram_counts[ngram] += 1

    models: list[NgramModel] = []
    for ngram, count in ngram_counts.items():
        model = NgramModel(
            dkey=doc.dkey,
            ngram=ngram,
            ngram_lc=ngram.lower(),
            ngram_tokens=len(ngram.split()),
            ngram_count=count,
            term_freq=1.0,
            doc_count=1,
        )
        models.append(model)
    return models


@pytest.fixture()
def papers():
    paper1 = PaperModel(
        dkey="doc1",
        meta_doi="doi",
        doi="doi",
        doc_doi="doi",
        doc_pub_date=datetime.utcnow(),
        pub_date=datetime.utcnow(),
        meta_pub_date=datetime.utcnow(),
        doc_author="authors",
        meta_author="authors",
        author="author",
        doc_title="title",
        meta_title="title",
        title="title",
    )
    paper2 = PaperModel(
        dkey="doc2",
        meta_doi="doi",
        doi="doi",
        doc_doi="doi",
        doc_pub_date=datetime.utcnow(),
        pub_date=datetime.utcnow(),
        meta_pub_date=datetime.utcnow(),
        doc_author="authors",
        meta_author="authors",
        author="author",
        doc_title="title",
        meta_title="title",
        title="title",
    )
    paper1_ngrams = make_ngrams(paper1, TEXT.strip())
    paper2_ngrams = make_ngrams(paper2, TEXT.strip())

    db.session.add_all([paper1, paper2, *paper1_ngrams, *paper2_ngrams])
    db.session.commit()
    return [paper1, paper2]
