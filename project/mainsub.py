"""
Dynamic turn‑taking debate between multiple **Llama‑cpp** agents (streaming)
===========================================================================
**v4 – モード分離とラベル固定**
--------------------------------
⚙️ **改良点**
1. **話者モードと聴取モードを完全分離**  
   *話す直前* にシステムメッセージ `SPEAKER_MODE` を一時挿入し、割り込み用 JSON を出さないよう強制。
2. **`Answer:` 行は丸ごと、`Reason:` ラベルも丸ごと** 履歴へ送信。  
   チャンク分割は *ラベル以降の本文* のみ。
3. **チャンク 3 語判定は同じ**（`WORDS_PER_CHUNK = 3`）。

```bash
$ python mainsub.py "Which option (A/B/C/D) is best?"
```
"""
from __future__ import annotations

import argparse
import json
import re
import time
from typing import List, Optional, Tuple

from llama_cpp import Llama
import config  # must expose get_model_path()

# ──────────────────────────────────────────────────────────────────────
WORDS_PER_CHUNK = 3
MAX_CHUNKS_PER_SPEECH = 30
INTERRUPT_KEYWORD = "INTERRUPT"
WAIT_KEYWORD = "WAIT"
CONCLUDE_KEYWORD = "CONCLUDE"
CTX_LIMIT_CHARS = 4096

SYSTEM_MSG = (
    "You are a helpful assistant engaged in a formal academic debate. "
    "Always follow the additional rules provided in system messages."
)

INTERRUPT_POLICY = (
    "Interrupt-policy:\n"
    "  • After every chunk of three words you hear from another agent, decide whether to interrupt.\n"
    "  • Respond ONLY with {\"decision\": \"INTERRUPT\"} or {\"decision\": \"WAIT\"}.\n"
    "  • Output nothing else."
)

SPEECH_FORMAT_POLICY = (
    "Speaking-format:\n"
    "  When it is your turn, reply exactly:\n"
    "    Answer: <one-line answer>\n"
    "    Reason: <concise reasoning>."
)

SPEAKER_MODE = (
    "You are now the SPEAKER. Ignore the interrupt-policy. Respond ONLY using the Speaking-format. "
    "Do NOT output any JSON such as {\"decision\": ...}."
)

PERSONAS = {
    "agent_A": "You are highly Analytical and always provide detailed evidence.",
    "agent_B": "You are highly Creative and favor out-of-the-box ideas.",
    "agent_C": "You are highly Critical but fair, focusing on logical consistency.",
}

# ──────────────────────────────────────────────────────────────────────
WORD_RE = re.compile(r"\S+")

def consume_words(buffer: str, n_words: int = WORDS_PER_CHUNK) -> Tuple[Optional[str], str]:
    if not buffer.strip():
        return None, buffer
    last_end = None
    for i, m in enumerate(WORD_RE.finditer(buffer), 1):
        if i == n_words:
            last_end = m.end()
            break
    if last_end is None:
        return None, buffer
    return buffer[:last_end], buffer[last_end:].lstrip()

# ──────────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, name: str, persona: str, model_path: str):
        self.name = name
        self.llm = Llama(
            model_path=model_path,
            chat_format="llama-3",
            n_ctx=16384,
            n_threads=8,
            n_gpu_layers=-1,
            verbose=False,
        )
        self.history: List[dict] = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "system", "content": persona},
            {"role": "system", "content": INTERRUPT_POLICY},
            {"role": "system", "content": SPEECH_FORMAT_POLICY},
        ]

    # ..................................................................
    def _trim(self):
        while sum(len(m["content"]) for m in self.history) > CTX_LIMIT_CHARS:
            for i, msg in enumerate(self.history):
                if msg["role"] != "system":
                    del self.history[i]
                    break

    # ..................................................................
    def stream_reply(self, prompt: str):
        self.history.append({"role": "user", "content": prompt})
        return self.llm.create_chat_completion(
            messages=self.history,
            stream=True,
            max_tokens=512,
            temperature=0.7,
        )

    # ..................................................................
    def wants_interrupt(self, chunk: str) -> bool:
        probe = f"The speaker emitted: «{chunk}»"
        resp = self.llm.create_chat_completion(
            messages=self.history + [{"role": "user", "content": probe}],
            max_tokens=12,
            temperature=0.0,
        )
        try:
            return json.loads(resp["choices"][0]["message"]["content"].strip()).get("decision") == INTERRUPT_KEYWORD
        except json.JSONDecodeError:
            return False

# ──────────────────────────────────────────────────────────────────────
class Debate:
    def __init__(self, model_path: str, personas: dict[str, str], topic: str):
        self.agents = [Agent(n, p, model_path) for n, p in personas.items()]
        self.topic = topic
        self.log: List[dict] = []

    # --------------------------------------------------------------
    def broadcast(self, speaker: str, content: str):
        msg = {"role": "assistant", "name": speaker, "content": content}
        self.log.append(msg)
        for ag in self.agents:
            ag.history.append(msg)
            ag._trim()

    # --------------------------------------------------------------
    def _prompt(self, first: bool, last_stmt: str = "") -> str:
        if first:
            return f"Debate topic: {self.topic}. Provide your initial answer and reasoning."
        return (
            "Previous speaker said:\n---\n" + last_stmt + "\n---\nRespond accordingly. If finished, start answer with '" + CONCLUDE_KEYWORD + "'."
        )

    # --------------------------------------------------------------
    def _first_speaker(self) -> int:
        for i, ag in enumerate(self.agents):
            vote = ag.llm.create_chat_completion(
                messages=ag.history + [{
                    "role": "user",
                    "content": "If you want to speak first, reply BID else PASS."
                }],
                max_tokens=4,
                temperature=0.0,
            )["choices"][0]["message"]["content"].strip().upper()
            if vote == "BID":
                return i
        return 0

    # --------------------------------------------------------------
    def run(self):
        idx = self._first_speaker()
        turn = 1
        while True:
            spk = self.agents[idx]
            print(f"\n==== Turn {turn}: {spk.name} ====")

            # ---- SPEAKER MODE injection ----
            spk.history.append({"role": "system", "content": SPEAKER_MODE})

            stream = spk.stream_reply(self._prompt(turn == 1, self.log[-1]["content"] if self.log else ""))

            buf = ""
            answer_sent = False
            reason_label_sent = False
            chunk_ct = 0
            interrupted = False

            for pkt in stream:
                delta = pkt["choices"][0]["delta"]
                if "content" not in delta:
                    continue
                buf += delta["content"]

                # ---------- Answer line ----------
                if not answer_sent and "\n" in buf:
                    ans, buf = buf.split("\n", 1)
                    ans = ans.strip()
                    if ans:
                        print(ans)
                        self.broadcast(spk.name, ans)
                        answer_sent = True

                # ---------- Reason label ----------
                if answer_sent and not reason_label_sent and buf.startswith("Reason:"):
                    label_len = len("Reason:")
                    print("Reason:")
                    self.broadcast(spk.name, "Reason:")
                    buf = buf[label_len:].lstrip()
                    reason_label_sent = True

                # ---------- Chunked Reason body ----------
                if reason_label_sent:
                    while True:
                        seg, buf = consume_words(buf, WORDS_PER_CHUNK)
                        if seg is None:
                            break
                        print(seg)
                        self.broadcast(spk.name, seg)
                        chunk_ct += 1
                        for j, lst in enumerate(self.agents):
                            if j == idx:
                                continue
                            if lst.wants_interrupt(seg):
                                print(f"\n>>> {lst.name} INTERRUPTS!\n")
                                stream.close()
                                interrupted = True
                                idx = j
                                break
                        if interrupted or chunk_ct >= MAX_CHUNKS_PER_SPEECH:
                            stream.close()
                            break
                if interrupted or chunk_ct >= MAX_CHUNKS_PER_SPEECH:
                    break

            # Flush leftover (rare)
            if not interrupted and buf.strip():
                self.broadcast(spk.name, buf.strip())
                print(buf.strip())

            # Remove SPEAKER_MODE directive
            spk.history.pop()

            # End condition
            if self.log and self.log[-1]["content"].startswith(f"Answer: {CONCLUDE_KEYWORD}"):
                print("\n### Debate concluded. ###")
                break

            if not interrupted:
                idx = (idx + 1) % len(self.agents)
            turn += 1
            time.sleep(0.05)

# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Multi‑agent streaming debate (Answer fixed, Reason chunked).")
    p.add_argument("topic", nargs="?", default="Which option (A/B/C/D) is best?", help="Debate topic question")
    args = p.parse_args()

    Debate(config.get_model_path(), PERSONAS, args.topic).run()


if __name__ == "__main__":
    main()
