from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class PretrainedModel(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    base_model_id: str
    model_id: str
    model_type: str
    path: str
    status: int
    create_time: datetime

    def _getattr(self, name):
        value = getattr(self, name)
        if isinstance(value, datetime):
            return value.timestamp()
        return value

