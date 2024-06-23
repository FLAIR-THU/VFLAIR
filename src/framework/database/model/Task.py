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
    params: Optional[str]
    create_time: datetime
    start_time: Optional[datetime]
    end_time: Optional[datetime]

    def to_dict(self):
        return {c.name: self._getattr(c.name) for c in self.__table__.columns}

    def _getattr(self, name):
        value = getattr(self, name)
        if isinstance(value, datetime):
            return value.timestamp()
        return value
