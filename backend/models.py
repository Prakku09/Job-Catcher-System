from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from .database import Base

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    company = Column(String, index=True)
    location = Column(String, index=True)
    description = Column(Text)
    url = Column(String, unique=True, index=True)
    source = Column(String)
    published_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
