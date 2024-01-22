import framework.database.model.Job as Job
from sqlmodel import Session, select

from framework.database.sql.engine import engine

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

job_repository = JobRepository()

