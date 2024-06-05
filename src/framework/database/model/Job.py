from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime


class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    fl_type: str
    params: str = Field(max_length=4000)
    status: int = Field(default=0)
    result: Optional[str] = Field(max_length=500)
    create_time: datetime
    end_time: Optional[datetime]

    def to_dict(self):
        return {c.name: self._getattr(c.name) for c in self.__table__.columns}

    def _getattr(self, name):
        value = getattr(self, name)
        if isinstance(value, datetime):
            return value.timestamp()
        return value
