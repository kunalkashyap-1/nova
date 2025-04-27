from calendar import c
import os 
from dotenv import load_dotenv

load_dotenv()


class Settings:
    DATABASE_URL = os.getenv("DATABASE_URL","127.0.0.1")
    SECRET_KEY = os.getenv("SECRET_KEY","dkslaflknui")
    ALGORITHM = os.getenv("ALGORITHM","HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES",30)
    CHROMA_URL = os.getenv("CHROMA_URL","http://chroma:8000") 


settings = Settings()