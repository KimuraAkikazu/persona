"""エージェントの思考と発言生成モジュール"""

from __future__ import annotations

from typing import Optional

from llm_client import LLMClient
from models import ActionDeclaration
from shared_memory import SharedMemory

class Agent:
    # ------------------------------
    # lifecycle
    # ------------------------------
    def __init__(self, agent_id: str, persona: str, memory: SharedMemory, llm: LLMClient):
        self.agent_id = agent_id
        self.persona = persona
        self.memory = memory
        self.llm = llm
        self.spoken_turns: int = 0

    # ------------------------------
    # prompts
    # ------------------------------
    _system_decl = """あなたは次の人物です:\n{persona}\n必ず JSON (utf-8) で出力してください。構造以外の文字列は禁止です。"""

    _user_decl = """現在の議論ログ:\n---\n{history}\n---\n行動を宣言してください。スキーマ: {{\"agent_id\": str, \"action_type\": \"speak|interrupt|listen\", \"urgency_score\": 0-1, \"summary\": str}}\n# JSON 例:\n{{\n  \"agent_id\": \"{agent_id}\",\n  \"action_type\": \"speak\",\n  \"urgency_score\": 0.8,\n  \"summary\": \"週休3日制の生産性向上事例を紹介したい\"\n}}"""

    _system_speak = """あなたは次の人物です:\n{persona}"""

    _user_speak = """議論トピック: {topic}\nこれまでのログ:\n---\n{history}\n---\nあなたの次の発言を日本語で 300 字以内で生成してください。"""

    # ------------------------------
    # public API
    # ------------------------------
    def think_and_declare(self, topic: str) -> ActionDeclaration:
        history = self.memory.get_latest_transcript_as_str(20)
        sys = self._system_decl.format(persona=self.persona)
        usr = self._user_decl.format(history=history, agent_id=self.agent_id)
        return self.llm.generate_structured_json(sys, usr, ActionDeclaration)

    def generate_utterance(self, topic: str) -> str:
        history = self.memory.get_latest_transcript_as_str(20)
        sys = self._system_speak.format(persona=self.persona)
        usr = self._user_speak.format(topic=topic, history=history)
        text = self.llm._chat([
            {"role": "system", "content": sys},
            {"role": "user", "content": usr},
        ])
        self.spoken_turns += 1
        return text.strip()