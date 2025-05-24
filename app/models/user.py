from sqlalchemy import Column, Integer, String, Text, DateTime, func
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    profile_picture = Column(Text, nullable=True)
    bio = Column(Text, default='')
    preferred_language = Column(String(50), default='')
    timezone = Column(String(100), default='')
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now()) 