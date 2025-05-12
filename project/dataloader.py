import pickle
from pathlib import Path


class dataloader:
    FILE_PATH = {
        "math": "./eval_data/math.pkl",
        "chess": "./eval_data/chess.pkl",
        "mmlu": "./eval_data/mmlu.pkl",
    }

    def __init__(self, name: str, n_case: int = 50):
        name = name.lower()
        assert name in self.FILE_PATH, f"dataset {name} is not valid."
        self.dataset = name
        self.n_case = n_case
        self.database = self._load_dataset()
        self.mode = "question"

    def _load_dataset(self):
        path = Path(self.FILE_PATH[self.dataset])
        print(f"data_path: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def set_mode(self, mode: str):
        assert mode in ["all", "question", "answer"], f"mode {mode} not valid."
        self.mode = mode

    def __len__(self):
        return min(self.n_case, len(self.database["task_info"]))

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("dataloader out of range")
        info = self.database["task_info"][idx]
        ans = self.database["answer"][idx]
        if self.mode == "question":
            return info
        elif self.mode == "answer":
            return ans
        else:
            return {"task_info": info, "answer": ans}
