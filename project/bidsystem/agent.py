import json
from llama_cpp import Llama

class Agent:
    """LLMを搭載したエージェント"""
    def __init__(self, name: str, model: Llama, system_prompt: str):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.debate_history = []

    def add_to_history(self, speaker: str, utterance: str):
        """議論の履歴に発言を追加する"""
        self.debate_history.append(f"{speaker}: {utterance}")

    def _create_messages(self, task_prompt: str) -> list:
        """LLMに渡すメッセージを作成する"""
        history_str = "\n".join(self.debate_history)
        
        user_prompt = f"""これまでの議論:
---
{history_str}
---
あなたのタスク:
{task_prompt}"""

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def bid(self, current_turn: int, max_turns: int) -> tuple[int, str]:
        """次の発言権を得るためのBid（入札）を行う"""
        task_prompt = f"""あなたは現在、議論のターン{current_turn}/{max_turns}にいます。
議論の状況を考慮し、次に発言したい意欲を0から3の数値で入札してください。
- 0: 静観したい。特に発言したいことはない。
- 1: 少し思うところがあり、発言したい。
- 2: 議論に貢献できる重要な意見がある。
- 3: 議論の流れを決定づける、絶対に発言すべきことがある。

以下のJSON形式で、あなたの思考とBid値を回答してください。
{{
  "reasoning": "（あなたの思考や判断理由をここに記述）",
  "bid": "（0から3の整数をここに記述）"
}}"""
        
        messages = self._create_messages(task_prompt)
        
        try:
            response = self.model.create_chat_completion(
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            result = json.loads(response['choices'][0]['message']['content'])
            bid_value = int(result.get("bid", 0))
            reasoning = result.get("reasoning", "")
            return bid_value, reasoning
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"エラー: {self.name}のBid処理に失敗しました。{e}")
            return 0, "エラーによりBidできませんでした。"

    def speak_chunked(self):
        """議論で発言する。ストリーム出力を文節ごとに分割して返すジェネレータ。"""
        task_prompt = "あなたの番です。これまでの議論を踏まえ、あなたの意見や考えを発言してください。"
        messages = self._create_messages(task_prompt)
        
        stream = self.model.create_chat_completion(
            messages=messages,
            temperature=0.7,
            stream=True,
        )
        
        buffer = ""
        # 区切り文字のセット
        delimiters = "。、」「！？"
        
        for output in stream:
            if "content" in output["choices"][0]["delta"]:
                token = output["choices"][0]["delta"]["content"]
                buffer += token
                
                # バッファ内に区切り文字が見つかったら、そこまでをyieldする
                while any(d in buffer for d in delimiters):
                    # 最初の区切り文字の位置を探す
                    first_delimiter_pos = float('inf')
                    for d in delimiters:
                        pos = buffer.find(d)
                        if pos != -1 and pos < first_delimiter_pos:
                            first_delimiter_pos = pos
                    
                    # 区切り文字を含めて文節を切り出す
                    chunk = buffer[:first_delimiter_pos + 1]
                    yield chunk
                    buffer = buffer[first_delimiter_pos + 1:]

        # ストリーム終了後、バッファに残った内容をyieldする
        if buffer:
            yield buffer