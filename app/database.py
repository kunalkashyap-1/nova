from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declartative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings


engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
sessionlocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declartative_base()

def get_db():
    db = sessionlocal()
    try:
        yield db
    finally:
        db.close()
        