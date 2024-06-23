import framework.database.model.Job as Job
from sqlmodel import Session, select
from datetime import datetime
from framework.database.sql.engine import engine
import json


class JobRepository:
    def create(self, job: Job.Job):
        with Session(engine) as session:
            session.add(job)
            session.commit()
            return job.id

    def get_by_id(self, id):
        with Session(engine) as session:
            statement = select(Job.Job).where(Job.Job.id == id)
            job = session.exec(statement).one()
            return job

    def save_state(self, id):
        with Session(engine) as session:
            statement = select(Job.Job).where(Job.Job.id == id)
            job = session.exec(statement).one()
            return job

    def change_status(self, job_id, status, result):
        with Session(engine) as session:
            statement = select(Job.Job).where(Job.Job.id == job_id)
            job = session.exec(statement).one()
            job.status = status
            job.end_time = datetime.now()
            if isinstance(result, str):
                job.result = result
            elif isinstance(result, object):
                job.result = json.dumps(result)
            session.add(job)
            session.commit()


job_repository = JobRepository()
