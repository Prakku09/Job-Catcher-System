from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class JobBase(BaseModel):
    title: str
    company: str
    location: str
    description: Optional[str] = None
    url: str
    source: str
    published_at: Optional[datetime] = None

class JobCreate(JobBase):
    pass

class Job(JobBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True
