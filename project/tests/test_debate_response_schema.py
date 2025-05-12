# tests/test_debate_response_schema.py

import pytest
import jsonschema
from agents import DEBATE_RESPONSE_SCHEMA, LlamaAgent


class DummyModel:
    def create_chat_completion(self, **kwargs):
        # schema に沿った応答を返す
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "function_call": {
                            "arguments": '{"reasoning":"test reasoning","answer":"C"}'
                        },
                    }
                }
            ]
        }


def test_schema_valid_instance():
    schema = DEBATE_RESPONSE_SCHEMA["parameters"]
    # 必須フィールドを満たすデータ
    data = {"reasoning": "foo", "answer": "A"}
    jsonschema.validate(instance=data, schema=schema)


def test_generate_response_matches_schema():
    model = DummyModel()
    agent = LlamaAgent("TestAgent", "", model, max_tokens=100)
    resp = agent.generate_response("dummy prompt")
    schema = DEBATE_RESPONSE_SCHEMA["parameters"]
    # jsonschema.validate が例外を出さなければ OK
    jsonschema.validate(instance=resp, schema=schema)
