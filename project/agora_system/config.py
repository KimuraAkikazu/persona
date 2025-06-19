from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
FILENAME = "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"

BASE_DIR = Path(__file__).parent.resolve()
MODEL_SAVE_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_SAVE_DIR / FILENAME

def get_model_path():
    """モデルをダウンロードし、ローカルパスを文字列として返す。"""
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"モデルをダウンロード中: {REPO_ID}/{FILENAME}")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=MODEL_SAVE_DIR,
            local_dir_use_symlinks=False,
        )
    print(f"モデルパス: {MODEL_PATH}")
    return str(MODEL_PATH)