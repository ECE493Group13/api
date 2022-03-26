import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
from typing import TYPE_CHECKING

import numpy as np
import word2vec_wrapper
from gensim.models import KeyedVectors
from logzero import logger
from sklearn.manifold import TSNE
from sqlalchemy.orm.session import Session

from api import app
from api.database import (
    DatasetModel,
    DatasetPaperModel,
    NgramModel,
    PaperModel,
    TrainedModel,
    TrainTaskModel,
    db,
)

if TYPE_CHECKING:
    from sqlalchemy.orm.query import Query


@contextmanager
def db_session():
    with app.app_context():
        session: Session = db.session
        yield session


def get_next_task(session: Session):
    task = (
        session.query(TrainTaskModel)
        .filter(TrainTaskModel.end_time.is_(None))
        .order_by(TrainTaskModel.created.asc())
        .limit(1)
        .one_or_none()
    )
    return task


def write_corpus(session: Session, dataset: DatasetModel, filename: Path):
    with open(filename, "w", encoding="utf-8") as file:
        query: "Query[NgramModel]" = (
            session.query(NgramModel)
            .join(PaperModel)
            .join(DatasetPaperModel)
            .join(DatasetModel)
            .filter(DatasetModel.id == dataset.id)
        )
        for ngram in query:
            file.write(f"{ngram.ngram_lc}\t{ngram.ngram_count}\n")


def read_embeddings(session: Session, task: TrainTaskModel, filename: Path):
    with open(filename, "rb") as file:
        model = TrainedModel(data=file.read(), task=task)
        session.add(model)


def generate_visualization(session: Session, task: TrainTaskModel, filename: Path):
    word_vectors = KeyedVectors.load_word2vec_format(filename, binary=False)

    vectors = np.array(word_vectors.vectors)
    labels = np.array(word_vectors.index_to_key)

    tsne = TSNE(n_components=2, random_state=0)

    vectors = tsne.fit_transform(vectors)

    x_points = [np.float64(vector[0]) for vector in vectors]
    y_points = [np.float64(vector[1]) for vector in vectors]

    data = {"labels": labels.tolist(), "x": x_points, "y": y_points}

    model: TrainedModel = (
        session.query(TrainedModel).filter(TrainedModel.task == task).one_or_none()
    )

    if model is None:
        print("No trained model found")
        return

    model.visualization = json.dumps(data)
    session.commit()


def run_task(session: Session, task: TrainTaskModel):
    hparams = json.loads(task.hparams)
    with TemporaryDirectory() as tempdir:
        corpus_filename = Path(tempdir) / "corpus.txt"
        embeddings_filename = Path(tempdir) / "embeddings.txt"

        logger.info("Write corpus to %s", corpus_filename)
        write_corpus(session, task.dataset, corpus_filename)
        logger.info("Train with hparams %s", hparams)
        word2vec_wrapper.train(corpus_filename, embeddings_filename, hparams)
        logger.info("Read corpus from %s", embeddings_filename)
        read_embeddings(session, task, embeddings_filename)
        generate_visualization(session, task, embeddings_filename)


def tick(session: Session):
    next_task = get_next_task(session)
    if next_task is None:
        return

    next_task.start_time = datetime.utcnow()
    session.commit()
    try:
        run_task(session, next_task)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        next_task.end_time = datetime.utcnow()
        session.commit()


def main():
    while True:
        with db_session() as session:
            try:
                tick(session)
            except Exception:  # pylint: disable=W0703
                logger.exception("Failed to run tick()")

        sleep(10)


if __name__ == "__main__":
    main()
