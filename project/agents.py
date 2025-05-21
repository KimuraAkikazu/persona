# project/agents.py
# ------------------------------------------------------------
# 変更点だけでなく **ファイル全体** を貼っています。コピペで上書き可
# ------------------------------------------------------------
import sys
import contextlib
import json
import re
from enum import Enum
from collections import deque
from typing import Deque, Literal, Optional

from pydantic import BaseModel
from llama_cpp import Llama

# ===== JSON schema (MMLU 用) =================================
DEBATE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answer":  {
            "type": "string",
            "enum": ["A", "B", "C", "D"],
        },
    },
    "required": ["reasoning", "answer"],
}
# =============================================================


class ChatTurn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OutputFormat(Enum):
    JSON = "json"
    PLAIN = "plain"


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open("/dev/null", "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def _clean_json_block(text: str) -> str:
    """```json ...``` などを削って最初の {...} を返す"""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    return m.group(0) if m else text


# ==================  LlamaAgent ==============================
class LlamaAgent:
    """
    persona を持つ 1 エージェント。
    * 1 問ごとに `reset_history()` が呼ばれる想定
    * deque(maxlen=12) で履歴を短く保持
    """

    def __init__(
        self,
        name: str,
        personality_text: str,
        model: Llama,
        max_tokens: int = 512,
    ):
        self.name = name
        self.personality_text = personality_text
        self.model = model
        self.max_tokens = max_tokens

        sys_msg = (
            f"You are {self.name}."
            if not personality_text
            else (
                f"Your personality traits:\n{personality_text}\n"
                "Answer concisely and stay consistent with them."
            )
        )
        self.system_turn = ChatTurn(role="system", content=sys_msg)

        # ── 会話履歴 ───────────────────────────────
        self.conversation_history: Deque[ChatTurn] = deque(
            [self.system_turn], maxlen=12
        )

    # ---------------------------------------------------------
    # public
    # ---------------------------------------------------------
    def reset_history(self):
        self.conversation_history = deque([self.system_turn], maxlen=12)

    def generate_response(
        self,
        user_prompt: str,
        *,
        enforce_json: bool = True,           # True=MMLU / False=BFI
        max_tokens: Optional[int] = None,
    ) -> dict:
        """LLM 呼び出し & 結果を dict で返す"""
        self.conversation_history.append(ChatTurn(role="user", content=user_prompt))

        messages = [t.model_dump() for t in self.conversation_history]
        kwargs = dict(
            messages=messages,
            max_tokens=max_tokens or self.max_tokens,
            temperature=0.7,
            top_p=0.9,
            seed=-1,
        )
        if enforce_json:
            kwargs["response_format"] = {
                "type": "json_object",
                "schema": DEBATE_RESPONSE_SCHEMA,
            }

        with suppress_stdout_stderr():
            out = self.model.create_chat_completion(**kwargs)

        raw = (out["choices"][0]["message"]["content"] or "").strip()

        reasoning, answer = "", raw  # デフォルト

        if enforce_json:
            # ---------------- JSON パース -----------------
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    data = json.loads(_clean_json_block(raw))
                except Exception:
                    data = {}
            reasoning = data.get("reasoning", "")
            answer = data.get("answer", answer)

        # 履歴追加
        self.conversation_history.append(ChatTurn(role="assistant", content=raw))

        return {"reasoning": reasoning, "answer": answer}

    # ---------------------------------------------------------
    # BFI helper
    # ---------------------------------------------------------
    def get_bfi_score(self, question: str, idx: int, total: int) -> int:
        prompt = (
            f"***Remember your personality traits***: {self.personality_text}\n"
            f"Question {idx}/{total}: {question}\n"
            "Respond ONLY with an integer 1-5 "
            "(1 = strongly disagree, 5 = strongly agree)."
        )
        resp = self.generate_response(prompt, enforce_json=False)
        try:
            return int(resp["answer"])
        except Exception:
            return 0


# ==================  AgentTriad ※変更なし ===================
class AgentTriad:
    def __init__(self, agentX: LlamaAgent, agentY: LlamaAgent, agentZ: LlamaAgent):
        self.agents = [agentX, agentY, agentZ]
        self.round_responses = {}

    def conduct_discussion(self, topic_prompt, max_turns: int = 3):
        pass  # Runner 側で実装

    def get_final_consensus(self) -> str:
        final_round = self.round_responses.get(max(self.round_responses.keys()), {})
        extracted = {a: r.get("answer", "") for a, r in final_round.items()}
        votes = {}
        for v in extracted.values():
            if v:
                votes[v] = votes.get(v, 0) + 1
        return max(votes, key=votes.get) if votes else "N/A"
