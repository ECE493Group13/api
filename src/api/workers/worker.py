from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from time import sleep
from typing import Type

from logzero import logger
from sqlalchemy.orm.session import Session

from api import app
from api.database import db


class Worker(ABC):
    @property
    @abstractmethod
    def task_model(self) -> Type[db.Model]:
        pass

    @abstractmethod
    def execute(self, session: Session, task: db.Model):
        pass


@contextmanager
def db_session():
    with app.app_context():
        session: Session = db.session
        yield session


class WorkerRunner:
    def __init__(self, worker: Worker):
        self.worker = worker

    def _get_next_task(self, session: Session):
        task = (
            session.query(self.worker.task_model)
            .filter(self.worker.task_model.end_time.is_(None))
            .order_by(self.worker.task_model.created.asc())
            .limit(1)
            .one_or_none()
        )
        return task

    def _tick(self, session: Session):
        logger.info("Get next task")
        next_task = self._get_next_task(session)
        if next_task is None:
            logger.info("No tasks yet")
            return

        next_task.start_time = datetime.utcnow()
        session.commit()

        logger.info("Running task")
        try:
            self.worker.execute(session, next_task)
            session.commit()
        except Exception:
            logger.exception("Task failed with exception")
            session.rollback()
            raise
        finally:
            logger.info("Task completed or failed")
            next_task.end_time = datetime.utcnow()
            session.commit()

    def execute(self):
        while True:
            with db_session() as session:
                try:
                    self._tick(session)
                except Exception:  # pylint: disable=W0703
                    logger.exception("Failed to run tick()")

            sleep(1)
