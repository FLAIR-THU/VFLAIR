from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional

class Task(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: str
    job_id: int
    status: int
    party: str
    run: str
    result: Optional[str]
    create_time: datetime
    start_time: Optional[datetime]
    end_time: Optional[datetime]

    def to_dict(self):
        return {
            "id": self.id,
            "task_id": self.task_id,
            "job_id": self.job_id,
            "status": self.status,
            "party": self.party,
            "run": self.run
        }


