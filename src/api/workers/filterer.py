from logzero import logger
from sqlalchemy.orm.session import Session

from api.database import DatasetModel, FilterTaskModel
from api.workers.worker import Worker, WorkerRunner


class FilterWorker(Worker):
    @property
    def task_model(self):
        return FilterTaskModel

    def execute(self, session: Session, task: FilterTaskModel):
        dataset = DatasetModel(task=task)
        session.add(dataset)
        session.flush()

        result = session.execute(
            """
            insert into dataset_paper (dataset_id, dkey)
                select distinct :dataset_id, dkey from doc_keywords_0
                where keywords_lc = :keywords
            """,
            {"dataset_id": task.dataset.id, "keywords": task.keywords},
        )
        logger.info(
            "Filtered papers: inserted %s rows into dataset_paper", result.rowcount
        )


def main():
    WorkerRunner(FilterWorker()).execute()


if __name__ == "__main__":
    main()
