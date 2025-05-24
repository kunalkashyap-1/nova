from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models.conversation import Conversation
from app.schemas.conversation import ConversationCreate, ConversationUpdate, ConversationOut

router = APIRouter(prefix="/api/v1/conversations", tags=["Conversations"])

@router.get("/", response_model=List[ConversationOut])
def list_conversations(db: Session = Depends(get_db)):
    return db.query(Conversation).all()

@router.post("/", response_model=ConversationOut)
def create_conversation(convo: ConversationCreate, db: Session = Depends(get_db)):
    conversation = Conversation(**convo.dict())
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation

@router.patch("/{id}", response_model=ConversationOut)
def update_conversation(id: int, updates: ConversationUpdate, db: Session = Depends(get_db)):
    conversation = db.query(Conversation).filter(Conversation.id == id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    for key, value in updates.dict(exclude_unset=True).items():
        setattr(conversation, key, value)
    db.commit()
    db.refresh(conversation)
    return conversation 