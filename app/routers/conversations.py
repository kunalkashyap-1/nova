from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from app.database import get_db
from app.models.conversation import Conversation
from app.models.user import User
from app.schemas.conversation import ConversationCreate, ConversationUpdate, ConversationOut
from app.models.message import Message

router = APIRouter(prefix="/api/v1/conversations", tags=["Conversations"])

@router.get("/", response_model=List[ConversationOut])
def list_conversations(db: Session = Depends(get_db)):
    return db.query(Conversation).all()

@router.post("/", response_model=ConversationOut)
def create_conversation(convo: ConversationCreate, db: Session = Depends(get_db)):
    """Create a new conversation. If the provided `user_id` is missing, non-numeric,
    or references a non-existent user, a temporary *guest* user is created and
    the conversation is associated with it.
    """
    user: Optional[User] = None

    # Attempt to resolve an existing user if a user_id was supplied
    if convo.user_id is not None:
        try:
            uid = int(convo.user_id)  # may raise ValueError if 'guest' or other str
            user = db.query(User).filter(User.id == uid).first()
        except (ValueError, TypeError):
            # Provided id is not an integer → treat as guest
            pass

    # Create a unique guest user if necessary
    if user is None:
        guest_suffix = uuid.uuid4().hex[:8]
        user = User(
            full_name=f"Guest {guest_suffix}",
            email=f"guest_{guest_suffix}@temporary.com",
            username=f"guest_{guest_suffix}",
            password_hash="",  # guests have no password
            is_guest=True,
        )
        db.add(user)
        db.flush()  # get generated user.id

    # Now create the conversation linked to this user
    conversation = Conversation(
        title=convo.title,
        user_id=user.id,
        model_id=convo.model_id,
        system_prompt=convo.system_prompt,
        folder_id=convo.folder_id,
    )
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

@router.get("/messages/{conversation_id}", status_code=status.HTTP_200_OK, summary="Get conversation messages")
async def get_conversation_messages(conversation_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    Get all messages for a specific conversation.
    
    This endpoint retrieves messages in chronological order (oldest to newest).
    
    Path parameters:
    - conversation_id: ID of the conversation to retrieve messages for
    
    Returns:
        A JSON object with a 'messages' array containing all messages in the conversation
    """
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at.asc()).all()
    
    return {"messages": messages}