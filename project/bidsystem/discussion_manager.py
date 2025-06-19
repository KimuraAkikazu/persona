# discussion_manager.py
import random
from typing import List
from agent import Agent

class DiscussionManager:
    def __init__(self, agents: List[Agent], topic: str, max_turns: int = 10):
        self.agents = agents
        self.topic = topic
        self.max_turns = max_turns
        self.discussion_history = []

    def run_discussion(self):
        """Bidシステムを用いて議論を実行する"""
        print("="*50)
        print(f"議題: {self.topic}")
        print(f"最大ターン数: {self.max_turns}")
        print("="*50)
        print("議論を開始します。")

        for current_turn in range(1, self.max_turns + 1):
            print(f"\n--- ターン {current_turn}/{self.max_turns} ---")

            # 1. 各エージェントが入札
            bids = {}
            for agent in self.agents:
                bid_value, reasoning = agent.bid(
                    self.topic, self.discussion_history, current_turn, self.max_turns
                )
                bids[agent.name] = {"bid": bid_value, "reasoning": reasoning}
                print(f"{agent.name}のBid: {bid_value} (理由: {reasoning})")

            # 2. 発言者を決定
            max_bid = max(b["bid"] for b in bids.values())

            # ご要望: 全員が0をBidした場合、沈黙のターンとする
            if max_bid == 0:
                print("\n>> 全員が発言を見送りました。沈黙のターンです。 <<")
                self.discussion_history.append({
                    "name": "システム",
                    "utterance": "（誰も発言しませんでした）"
                })
                continue

            # 最高額入札者の中からランダムで発言者を決定
            highest_bidders = [
                name for name, data in bids.items() if data["bid"] == max_bid
            ]
            speaker_name = random.choice(highest_bidders)
            speaker_agent = next(agent for agent in self.agents if agent.name == speaker_name)
            bid_reasoning = bids[speaker_name]["reasoning"]

            print(f"\n発言者: {speaker_name} (Bid: {max_bid})")

            # 3. 発言
            utterance = speaker_agent.speak(
                self.topic, self.discussion_history, current_turn, self.max_turns, bid_reasoning
            )
            
            # 4. 議論履歴を更新
            self.discussion_history.append({"name": speaker_name, "utterance": utterance})

        print("\n--- 議論終了 ---")
        print("最終的な議論の履歴:")
        for log in self.discussion_history:
            print(f"- {log['name']}: {log['utterance']}")