"""議論全体の進行を制御するマネージャ"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Dict

from agent import Agent
from models import ActionDeclaration
from shared_memory import SharedMemory

class DialogueManager:
    def __init__(self, agents: List[Agent], memory: SharedMemory, cfg: Dict):
        self.agents = agents
        self.memory = memory
        self.topic = cfg["discussion"]["topic"]
        self.max_turns = cfg["discussion"]["max_turns"]
        self.interrupt_th = cfg["arbitration"]["interrupt_urgency_threshold"]
        self.fairness = cfg["arbitration"]["fairness_factor"]
        self.turn = 0
        self.speak_counts = defaultdict(int)

    # ------------------------------
    # helpers
    # ------------------------------
    def _collect_declarations(self) -> List[ActionDeclaration]:
        return [a.think_and_declare(self.topic) for a in self.agents]

    def _arbitrate_speaker(self, decls: List[ActionDeclaration]) -> str | None:
        # 優先: 割り込み
        interrupts = [d for d in decls if d.action_type == "interrupt" and d.urgency_score >= self.interrupt_th]
        if interrupts:
            chosen = max(interrupts, key=lambda d: d.urgency_score)
            print(f"[Arbitrate] INTERRUPT => {chosen.agent_id} (urgency={chosen.urgency_score:.2f})")
            return chosen.agent_id

        # 通常発言
        speaks = [d for d in decls if d.action_type == "speak"]
        if not speaks:
            return None

        def score(d: ActionDeclaration) -> float:
            fairness_bonus = self.fairness * (1 / (1 + self.speak_counts[d.agent_id]))
            return d.urgency_score + fairness_bonus

        scored = [(d, score(d)) for d in speaks]
        for d, s in scored:
            print(f"[Arbitrate] {d.agent_id} base={d.urgency_score:.2f} fair+={s - d.urgency_score:.2f} => {s:.2f} | {d.summary}")
        chosen, _ = max(scored, key=lambda t: t[1])
        return chosen.agent_id

    # ------------------------------
    # public API
    # ------------------------------
    def run_discussion(self) -> None:
        print(f"=== Agora Discussion START ===\nTopic: {self.topic}\n")
        while self.turn < self.max_turns:
            self.turn += 1
            print(f"\n--- Turn {self.turn} ---")

            decls = self._collect_declarations()
            speaker_id = self._arbitrate_speaker(decls)

            if speaker_id is None:
                print("沈黙のターンです")
                continue

            speaker = next(a for a in self.agents if a.agent_id == speaker_id)
            utter = speaker.generate_utterance(self.topic)

            self.memory.add_utterance(self.turn, speaker_id, utter)
            self.speak_counts[speaker_id] += 1

            print(f"{speaker_id}: {utter}")

        print("\n=== Discussion Finished ===")