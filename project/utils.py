# utils.py
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens_tiktoken(text: str) -> int:
    """
    Return the number of tokens for given text using tiktoken.
    """
    return len(ENC.encode(text))
