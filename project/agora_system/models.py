from pydantic import BaseModel, Field
from typing import Literal

class ActionDeclaration(BaseModel):
    agent_id: str
    action_type: Literal["speak", "interrupt", "listen"]
    urgency_score: float = Field(ge=0.0, le=1.0)
    summary: str

class Utterance(BaseModel):
    turn: int
    agent_id: str
    content: str
    timestamp: str