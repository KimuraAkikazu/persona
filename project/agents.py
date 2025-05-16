# agents.py  (T3: ChatTurn + deque(maxlen=10))
import sys
import contextlib
from enum import Enum
from collections import deque
from typing import Deque, Literal  # List は未使用のため削除


from pydantic import BaseModel
from llama_cpp import Llama


# ---------- 型定義 ----------
class ChatTurn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OutputFormat(Enum):
    JSON = "json"
    PLAIN = "plain"


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open("/dev/null", "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def filter_repetitions(text: str, max_length: int = 200) -> str:
    lines = text.splitlines()
    filtered = [
        ln for ln in lines if not (len(ln) > max_length and "response is" in ln)
    ]
    return "\n".join(filtered)


# ---------- LlamaAgent ----------
class LlamaAgent:
    """
    各エージェントは ChatTurn(BaseModel) で履歴を保持する。
    deque(maxlen=10) により古い発話は自動でドロップされる。
    """

    def __init__(
        self, name: str, personality_text: str, model: Llama, max_tokens: int = 512
    ) -> None:
        self.name = name
        self.personality_text = personality_text
        self.model = model
        self.max_tokens = max_tokens

        sys_msg = (
            f"You are {self.name}."
            if personality_text == ""
            else (
                f"You are {self.name} with the following personality traits:\n"  # 長い行を改行
                f"{personality_text}\n"
                "Answer concisely and strictly adhere to these traits. "  # 長い行を改行
                "Ensure your reasoning reflects your personality. "
                "When answering multiple‑choice questions, output JSON with "  # 長い行を改行
                'keys "reasoning" and "answer".'
            )
        )
        self.system_turn = ChatTurn(role="system", content=sys_msg)

        # deque で履歴管理
        self.conversation_history: Deque[ChatTurn] = deque(
            [self.system_turn], maxlen=10
        )

    # ---- public API ----
    def reset_history(self) -> None:
        self.conversation_history = deque([self.system_turn], maxlen=10)

    def generate_response(self, user_prompt: str) -> dict:
        """ユーザ入力を追加し、LLM から assistant 応答を取得して返す"""
        self.conversation_history.append(
            ChatTurn(role="user", content=user_prompt)  # 長い行を改行
        )

        messages = [turn.model_dump() for turn in self.conversation_history]

        with suppress_stdout_stderr():
            output = self.model.create_chat_completion(  # 長い行を改行
                messages,
                max_tokens=self.max_tokens,
                temperature=0.7,
                top_p=0.9,
                stop=["System:", "User:"],
                seed=-1,
            )

        raw_text = output["choices"][0]["message"]["content"].strip()
        try:
            json_resp = (
                eval(raw_text) if raw_text.startswith("{") else {}  # 長い行を改行
            )
        except Exception:  # fallback
            json_resp = {}

        reasoning = json_resp.get("reasoning", "")
        answer = json_resp.get("answer", raw_text)

        self.conversation_history.append(
            ChatTurn(role="assistant", content=raw_text)  # 長い行を改行
        )

        return {"reasoning": reasoning, "answer": answer}
    def get_bfi_score(self, question: str, idx: int, total: int) -> int:
        """
        Big-Five 質問に 1-5 の整数で回答させる。
        戻り値がパースできない場合は 0 を返す。
        """
        prompt = (
            f"You are currently answering a personality inventory.\n"
            f"Question {idx}/{total}: {question}\n"
            "Respond **only** with a single integer between 1 and 5, "
            "where 1 means 'strongly disagree' and 5 means 'strongly agree'."
        )
        resp = self.generate_response(prompt)
        try:
            return int(resp["answer"])
        except Exception:
            return 0


# ---------- AgentTriad (変更なし) ----------
class AgentTriad:
    """
    3体のエージェントによるディベートを管理するクラス。
    """

    def __init__(self, agentX: LlamaAgent, agentY: LlamaAgent, agentZ: LlamaAgent):
        self.agents = [agentX, agentY, agentZ]
        self.round_responses = {}

    def conduct_discussion(self, topic_prompt, max_turns=3):
        # 省略: Runner 側で実装するため未使用
        pass

    def get_final_consensus(self):
        final_round = self.round_responses.get(max(self.round_responses.keys()), {})
        extracted = {}
        for agent_name, resp in final_round.items():
            ans = resp.get("answer", "") if isinstance(resp, dict) else ""
            extracted[agent_name] = str(ans).strip() or "N/A"
        votes = {}
        for v in extracted.values():
            if v != "N/A":
                votes[v] = votes.get(v, 0) + 1
        return max(votes, key=votes.get) if votes else "N/A"
