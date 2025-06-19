"""議論のログを保持する共有メモリ"""

from datetime import datetime
from typing import List

from models import Utterance

class SharedMemory:
    def __init__(self) -> None:
        self._log: List[Utterance] = []

    # ------------------------------
    # mutation
    # ------------------------------
    def add_utterance(self, turn: int, agent_id: str, content: str) -> None:
        self._log.append(
            Utterance(
                turn=turn,
                agent_id=agent_id,
                content=content,
                timestamp=datetime.utcnow().isoformat(),
            )
        )

    # ------------------------------
    # accessors
    # ------------------------------
    def get_transcript(self) -> List[Utterance]:
        return list(self._log)

    def get_latest_transcript_as_str(self, n: int | None = None) -> str:
        target = self._log[-n:] if n else self._log
        return "\n".join(f"[{u.turn}] {u.agent_id}: {u.content}" for u in target)