from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    fl_type: str
    params: str
    create_time: datetime

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

