"""LLM Client: 安定した構造化 JSON を取得するためのユーティリティ"""

import json
from pathlib import Path
from typing import Any, Dict, List, Type

import yaml
from llama_cpp import Llama

from config import BASE_DIR, get_model_path
from models import ActionDeclaration

class LLMClient:
    def __init__(self) -> None:
        # 設定読み込み
        with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.llm_params: Dict[str, Any] = cfg["llm_params"]
        self.gen_params: Dict[str, Any] = cfg["generation_params"]
        self.retry_attempts: int = cfg["llm_client"].get("json_retry_attempts", 3)

        # モデルロード
        model_path = get_model_path()
        self.llama = Llama(model_path=model_path, chat_format="llama-3", **self.llm_params)

    # ------------------------------
    # internal helpers
    # ------------------------------
    def _chat(self, messages: List[Dict[str, str]], *, temperature: float | None = None, max_tokens: int | None = None) -> str:
        """LLama.cpp のラッパー。単一テキストを返す"""
        temperature = temperature if temperature is not None else self.gen_params["temperature"]
        max_tokens = max_tokens if max_tokens is not None else self.gen_params["max_tokens"]
        response = self.llama.create_chat_completion(messages=messages, temperature=temperature, max_tokens=max_tokens)
        return response["choices"][0]["message"]["content"]

    # ------------------------------
    # public API
    # ------------------------------
    def generate_structured_json(self, system_prompt: str, user_prompt: str, schema: Type[ActionDeclaration]):
        """指定スキーマに従う JSON を返す。失敗時はリトライ。"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(1, self.retry_attempts + 1):
            raw = self._chat(messages)
            try:
                parsed = json.loads(raw)
                return schema.model_validate(parsed)
            except Exception as e:
                print(f"[LLMClient] JSON parse failed (attempt {attempt}/{self.retry_attempts}): {e}\nOutput was:\n{raw}\n")
                if attempt == self.retry_attempts:
                    raise
                # フィードバックを与えて再試行
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": "JSON が無効です。上記スキーマに合致する JSON のみを返してください。余計な文字列を含めないでください。",
                })