# prompts.py
from typing import List, Dict

def get_system_prompt(topic: str, max_turns: int) -> str:
    """エージェントの役割と議論のルールを定義するシステムプロンプト"""
    return f"""あなたは優秀な議論参加者です。他の参加者と協力して、与えられた議題について議論し、結論を導き出すことを目指します。

# 議題
{topic}

# ルール
- 議論は最大 {max_turns} ターンです。それまでに何らかの結論を出してください。
- 自分の意見を明確に述べ、他の人の意見も尊重してください。
- 感情的にならず、論理的な議論を心がけてください。
"""

def get_bidding_prompt(
    agent_name: str,
    topic: str,
    discussion_history: List[Dict[str, str]],
    current_turn: int,
    max_turns: int
) -> str:
    """次の発言権を得るためのBidを促すプロンプト"""

    history_str = "\n".join([f"{log['name']}: {log['utterance']}" for log in discussion_history])
    if not history_str:
        history_str = "まだ誰も発言していません。"

    return f"""あなたは {agent_name} です。現在の議論の状況を考慮し、次に発言したい度合いを0から3の数値で入札（Bid）してください。

# 議題
{topic}

# 現在の状況
- 現在 {current_turn} ターン目です（最大 {max_turns} ターン）。

# これまでの議論
{history_str}

# Bidの選択肢
- 0: 今は静観し、他の人の話を聞きたい。
- 1: いくつか一般的な考えを共有したい。
- 2: 議論に貢献できる、重要かつ具体的なことがある。
- 3: 議論の方向性を変えるような、絶対に言うべきことがある。

# あなたの思考プロセスとBidをJSON形式で出力してください。
例:
{{
  "reasoning": "（ここにあなたの思考を記述）",
  "bid": 1/2/3/4
}}
"""

def get_speaking_prompt(
    agent_name: str,
    topic: str,
    discussion_history: List[Dict[str, str]],
    current_turn: int,
    max_turns: int,
    bid_reasoning: str
) -> str:
    """発言を生成するためのプロンプト"""

    history_str = "\n".join([f"{log['name']}: {log['utterance']}" for log in discussion_history])
    if not history_str:
        history_str = "まだ誰も発言していません。"

    return f"""あなたは {agent_name} です。あなたの発言ターンです。
これまでの議論の流れと、あなたが入札時に考えた「{bid_reasoning}」という理由を踏まえて、議題に対する意見や質問などを具体的に発言してください。

# 議題
{topic}

# 現在の状況
- 現在 {current_turn} ターン目です（最大 {max_turns} ターン）。

# これまでの議論
{history_str}

# あなたの発言
（ここにあなたの発言を記述）
"""