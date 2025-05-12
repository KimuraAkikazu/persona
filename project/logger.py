# logger.py
import os
import json
import csv
import yaml
import datetime
import subprocess
from pathlib import Path
from typing import Dict, Any


class ExperimentLogger:
    def __init__(
        self, team: str, seed: int, model_path: str, base_dir: Path = Path("results")
    ) -> None:
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.team = team
        self.seed = seed
        self.model_path = model_path
        self.dir = base_dir / self.run_id / team
        self.dir.mkdir(parents=True, exist_ok=True)

        # run_config.yaml の初期書き込み
        run_dir = self.dir.parent
        cfg_path = run_dir / "run_config.yaml"
        if not cfg_path.exists():
            self._write_run_config(run_dir)

        self.jsonl_path = self.dir / "debate.jsonl"
        self.metrics_path = self.dir / "metrics.csv"
        self._init_metrics_csv()

    def log_turn(self, record: Dict[str, Any]) -> None:
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    def log_metric(self, row: Dict[str, Any]) -> None:
        with open(self.metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._headers)
            writer.writerow(row)

    def _init_metrics_csv(self) -> None:
        self._headers = [
            "q_id",
            "final_answer",
            "correct_answer",
            "correct",
            "total_tokens",
        ]
        if not self.metrics_path.exists():
            with open(self.metrics_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._headers)
                writer.writeheader()

    def _write_run_config(self, run_dir: Path) -> None:
        cfg: Dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode()
            .strip(),
            "model_path": self.model_path,
            "python_version": subprocess.check_output(["python", "-V"])
            .decode()
            .strip(),
            "seed": self.seed,
        }
        with open(run_dir / "run_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
