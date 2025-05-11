from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from app.database import get_db
from app.schemas.memory import MemoryCreate, MemoryOut
from app.models.memory import Memory
import chromadb 
from chromadb.config import Settings
from typing import List
import os
from dotenv import load_dotenv

router = APIRouter()
load_dotenv()


MAX_MEMORIES = 200

def get_chroma_client():
    env_path = os.getenv("CHROMA_DB_PATH", "../storage/chromadb")
    chroma_client = chromadb.PersistentClient(path=env_path)
    return chroma_client

@router.post("/", response_model=MemoryOut)
def create_memory(memory: MemoryCreate, chroma_client=Depends(get_chroma_client), user_id: str = ""):
    # count existing memories
    # if the count is more than max Remove the oldest memory
    # create generate a consice memory from the prompt given 
    # generate an embedding for the generated memory
    # store the memory in chroma
    # return the memory
    return


@router.get("/", response_model=List[MemoryOut])
def get_memories(chroma_client=Depends(get_chroma_client), user_id: str = "") -> List[MemoryOut]:
    # get all memories from chroma
    # return the memories as a list of MemoryOut objects
    pass