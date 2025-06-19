"""エントリーポイント"""

import yaml
from pathlib import Path

from agent import Agent
from dialogue_manager import DialogueManager
from llm_client import LLMClient
from shared_memory import SharedMemory
from config import BASE_DIR


def main():
    # 設定ロード
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # コンポーネント初期化
    llm_client = LLMClient()
    memory = SharedMemory()

    agents = [
        Agent(a_cfg["agent_id"], a_cfg["persona"], memory, llm_client)
        for a_cfg in cfg["agents"]
    ]

    manager = DialogueManager(agents, memory, cfg)
    manager.run_discussion()


if __name__ == "__main__":
    main()