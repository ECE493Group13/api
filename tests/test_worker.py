from datetime import datetime

import pytest
from flask.testing import FlaskClient
from sqlalchemy import Column, DateTime, Integer
from sqlalchemy.orm.session import Session

from api.database import db
from api.workers.worker import Worker, WorkerRunner


@pytest.fixture(autouse=True)
def db_session(client: FlaskClient):
    with client.application.app_context():
        yield db.session


class MockTaskModel(db.Model):
    id = Column(Integer, primary_key=True)
    created = Column(DateTime, nullable=False, default=datetime.utcnow)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    result = Column(Integer, nullable=True)


class SuccessWorker(Worker):
    @property
    def task_model(self):
        return MockTaskModel

    def execute(self, session: Session, task: MockTaskModel):
        task.result = 1


class FailWorker(Worker):
    @property
    def task_model(self):
        return MockTaskModel

    def execute(self, session: Session, task: MockTaskModel):
        raise RuntimeError("something happened!")


class TestWorkerRunner:
    def test_tick_success(self, db_session: Session):
        """
        WorkerRunner._tick() calls Worker.execute() and sets start/end time.
        """
        task = MockTaskModel()
        db_session.add(task)
        db_session.commit()

        WorkerRunner(SuccessWorker())._tick(db_session)  # pylint: disable=W0212

        assert task.start_time is not None
        assert task.end_time is not None
        assert task.result == 1

    def test_tick_fail(self, db_session: Session):
        """
        If worker.execute() fails, WorkerRunner._tick() still sets start/end
        time.
        """
        task = MockTaskModel()
        db_session.add(task)
        db_session.commit()

        with pytest.raises(RuntimeError):
            WorkerRunner(FailWorker())._tick(db_session)  # pylint: disable=W0212

        assert task.start_time is not None
        assert task.end_time is not None
        assert task.result is None
