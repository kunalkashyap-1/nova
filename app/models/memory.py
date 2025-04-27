from sqlalchemy import Column, Integer, String, DateTime
from app.database import Base
from datetime import datetime

class Memory(Base):
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
