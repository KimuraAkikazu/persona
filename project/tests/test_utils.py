# tests/test_utils.py

import pytest
from utils import count_tokens_tiktoken


def test_count_tokens_empty():
    assert count_tokens_tiktoken("") == 0


def test_count_tokens_simple():
    text = "Hello world"
    # tiktoken 実装次第のため、必ず 2 トークン以上を想定
    tokens = count_tokens_tiktoken(text)
    assert isinstance(tokens, int)
    assert tokens >= 2
