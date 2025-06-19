import time
import random
from typing import List
from agent import Agent

class DebateModerator:
    """Bidシステムを用いて議論を進行させるモデレーター"""
    def __init__(self, agents: List[Agent], topic: str, max_turns: int = 15):
        self.agents = agents
        self.topic = topic
        self.max_turns = max_turns
        self.current_turn = 1
        self.global_debate_log = []
        
        # 割り込み可能な議論のための状態変数
        self.current_speaker = None
        self.speaker_utterance_generator = None

    def _broadcast(self, speaker_name: str, message_chunk: str):
        """全エージェントに発言の断片を共有する"""
        for agent in self.agents:
            agent.add_to_history(speaker_name, message_chunk)
        self.global_debate_log.append(f"{speaker_name}: {message_chunk}")

    def run_debate(self):
        """議論のメインループを実行する"""
        print("--- 議論開始 ---")
        print(f"議題: {self.topic}")
        print(f"ルール: {self.max_turns}ターン以内に結論を出してください。")
        print("-" * 20 + "\n")

        # システムプロンプトを各エージェントに設定
        system_prompt = f"あなたは{len(self.agents)}人で行われる議論の参加者です。議題は「{self.topic}」です。他の参加者と協力し、{self.max_turns}ターン以内に結論を出してください。"
        for agent in self.agents:
            agent.system_prompt = system_prompt

        while self.current_turn <= self.max_turns:
            print(f"\n--- ターン {self.current_turn}/{self.max_turns} ---")

            # --- 1. 割り込み/発言権の入札 (Bidding) ---
            bids = {}
            for agent in self.agents:
                # 発言中のエージェントは自動的に最高Bidを持つ（継続の意思）
                if self.current_speaker and agent.name == self.current_speaker.name:
                    bids[agent.name] = 4 # 継続のための特別Bid
                else:
                    bid, reasoning = agent.bid(self.current_turn, self.max_turns)
                    bids[agent.name] = bid
                    print(f"  [Bid] {agent.name}: {bid} (理由: {reasoning})")

            max_bid = max(bids.values())

            # --- 2. 発言者の決定 ---
            if max_bid == 0:
                print("\n>> 全員が静観を選択しました。沈黙のターンです。 <<")
                self.current_speaker = None
                self.speaker_utterance_generator = None
                self.current_turn += 1
                time.sleep(2)
                continue

            highest_bidders = [name for name, bid in bids.items() if bid == max_bid]
            next_speaker_name = random.choice(highest_bidders)
            
            # --- 3. 発言の実行と割り込み処理 ---
            # スピーカーが交代した場合（割り込み発生）
            if not self.current_speaker or next_speaker_name != self.current_speaker.name:
                if self.current_speaker:
                    print(f"\n>> {next_speaker_name}が{self.current_speaker.name}の発言に割り込みます！ <<")
                
                self.current_speaker = next((agent for agent in self.agents if agent.name == next_speaker_name), None)
                self.speaker_utterance_generator = self.current_speaker.speak_chunked()
                print(f"\n発言者: {self.current_speaker.name}")

            # --- 4. 発言内容（1文節）の出力 ---
            try:
                chunk = next(self.speaker_utterance_generator)
                print(chunk, end="", flush=True)
                # 全員に発言の断片を共有
                self._broadcast(self.current_speaker.name, chunk)

            except StopIteration:
                # 発言者が自然に発言を終えた場合
                print("\n(発言終了)")
                self.current_speaker = None
                self.speaker_utterance_generator = None
            
            self.current_turn += 1
            time.sleep(0.5)

        print("\n--- 議論終了 ---")
        print("最終的な議論のログ:")
        print("\n".join(self.global_debate_log))