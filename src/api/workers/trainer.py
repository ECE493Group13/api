import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import word2vec_wrapper
from gensim.models import KeyedVectors
from gensim.models.phrases import Phraser, Phrases
from logzero import logger
from sklearn.manifold import TSNE
from sqlalchemy.orm.session import Session

from api.database import (
    DatasetModel,
    DatasetPaperModel,
    NgramModel,
    TrainedModel,
    TrainTaskModel,
)
from api.workers.worker import Worker, WorkerRunner

if TYPE_CHECKING:
    from sqlalchemy.orm.query import Query


DATA_ROOT_PATH = Path(__file__).parent.parent.parent.parent / "fs"


def write_corpus(session: Session, dataset: DatasetModel, filename: Path):
    with open(filename, "w", encoding="utf-8") as file:
        query: "Query[NgramModel]" = (
            session.query(NgramModel)
            .join(DatasetPaperModel, DatasetPaperModel.dkey == NgramModel.dkey)
            .filter(DatasetPaperModel.dataset_id == dataset.id)
            .all()
        )

        phrases = Phrases(sentences=None, min_count=20, threshold=5, progress_per=1000)
        for ngram in query:
            phrases.add_vocab([ngram.ngram_lc.split(" ")])

        phrases_model = Phraser(phrases)

        for ngram in query:
            processed_ngram = " ".join(phrases_model[ngram.ngram_lc.split(" ")])
            file.write(f"{processed_ngram}\t{ngram.ngram_count}\n")


def generate_visualization(embeddings_filename: Path):
    word_vectors = KeyedVectors.load_word2vec_format(embeddings_filename, binary=False)

    vectors = np.array(word_vectors.vectors)
    labels = np.array(word_vectors.index_to_key)

    tsne = TSNE(n_components=2, random_state=0)

    vectors = tsne.fit_transform(vectors)

    x_points = [np.float64(vector[0]) for vector in vectors]
    y_points = [np.float64(vector[1]) for vector in vectors]

    data = {"labels": labels.tolist(), "x": x_points, "y": y_points}

    return json.dumps(data)


def save_model(
    session: Session, task: TrainTaskModel, embeddings_filename: Path, visualization
):
    model = TrainedModel(
        task=task,
        embeddings_filename=str(embeddings_filename),
        visualization=visualization,
    )
    session.add(model)
    session.commit()
    return model


def run_task(session: Session, task: TrainTaskModel, data_root: Path):
    data_root.mkdir(exist_ok=True)

    hparams = json.loads(task.hparams)
    with TemporaryDirectory() as tempdir:
        corpus_filename = Path(tempdir) / "corpus.txt"
        embeddings_filename = data_root / f"embeddings_{datetime.utcnow()}_{uuid4()}"

        logger.info("Write corpus to %s", corpus_filename)
        write_corpus(session, task.dataset, corpus_filename)
        logger.info(
            "Train with hparams %s and save embeddings to %s",
            hparams,
            embeddings_filename,
        )
        word2vec_wrapper.train(corpus_filename, embeddings_filename, hparams)
        logger.info("Generate visualization from %s", embeddings_filename)
        visualization = generate_visualization(embeddings_filename)
        save_model(session, task, embeddings_filename, visualization)


class TrainWorker(Worker):
    def __init__(self, data_root: Path):
        self.data_root = data_root

    @property
    def task_model(self):
        return TrainTaskModel

    def execute(self, session: Session, task: TrainTaskModel):
        run_task(session, task, self.data_root)


def main():
    WorkerRunner(TrainWorker(data_root=DATA_ROOT_PATH)).execute()


if __name__ == "__main__":
    main()
