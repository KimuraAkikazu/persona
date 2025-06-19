# llm_client.py
import json
from llama_cpp import Llama
from config import get_model_path

class LlamaClient:
    def __init__(self, model_path, n_gpu_layers=-1, n_ctx=8192):
        """Llama.cppモデルをロードしてクライアントを初期化する"""
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers, # GPUにオフロードするレイヤー数 (-1はすべて)
            n_ctx=n_ctx,               # コンテキストサイズ
            verbose=False
        )

    def generate_json(self, prompt: str) -> dict:
        """プロンプトからJSONレスポンスを生成する"""
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that always responds in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_object",
            },
            temperature=0.7,
        )
        try:
            return json.loads(response['choices'][0]['message']['content'])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"JSONのパースに失敗しました: {e}")
            print(f"RAWレスポンス: {response['choices'][0]['message']['content']}")
            return {}

    def generate_stream(self, prompt: str, system_prompt: str):
        """プロンプトから発言をストリーミングで生成する"""
        stream = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            temperature=0.8,
        )
        for output in stream:
            content = output['choices'][0]['delta'].get('content', None)
            if content:
                yield content