# models.py
from typing import Literal
from pydantic import BaseModel


class ChatTurn(BaseModel):
    """
    1 度の発話を表す構造体。
    Validation・シリアライズは Pydantic が自動で実行。
    """

    role: Literal["system", "user", "assistant"]
    content: str
