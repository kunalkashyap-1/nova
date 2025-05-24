from sqlalchemy import Column, Integer, String, Text, Boolean, Numeric, JSON, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    provider = Column(String(100), nullable=False)  # e.g., OpenAI, Anthropic, etc.
    model_id = Column(String(100), nullable=False)  # The ID used by the provider API
    description = Column(Text)
    capabilities = Column(JSON)  # Store capabilities as JSON
    parameters = Column(JSON)  # Default parameters as JSON
    is_active = Column(Boolean, default=True)
    cost_per_1k_input_tokens = Column(Numeric(10, 6))
    cost_per_1k_output_tokens = Column(Numeric(10, 6))
    max_tokens = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
