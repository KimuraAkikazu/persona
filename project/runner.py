# runner.py
from pathlib import Path
from typing import List, Dict, Any
from jinja2 import Template
from llama_cpp import Llama
from agents import LlamaAgent, AgentTriad
from dataloader import dataloader
from logger import ExperimentLogger
from utils import count_tokens_tiktoken
from bfi import BFI_QUESTIONS, compute_bfi_scores
import json
import pandas as _pd





# テンプレート読み込み
def _load_tpl(name: str) -> Template:
    return Template((Path("templates") / name).read_text(encoding="utf-8"))


_TPL_R1 = _load_tpl("round1.j2")
_TPL_RN = _load_tpl("roundN.j2")


class ExperimentRunner:
    def __init__(
        self,
        llama: Llama,
        bigfive_prompts: Dict[str, str],
        n_case: int,
        seed: int,
        model_path: str,
        run_id: str
    ):
        self.llama = llama
        self.prompts = bigfive_prompts
        self.n_case = n_case
        self.seed = seed
        self.model_path = model_path
        self.run_id = run_id

    def run_team(self, team_name: str, personalities: List[str]) -> None:
        logger = ExperimentLogger(
            team=team_name, seed=self.seed, model_path=self.model_path,run_id=self.run_id
        )

        agents = [
            LlamaAgent(f"Agent{i+1}", personalities[i], self.llama, max_tokens=1024)
            for i in range(3)
        ]
        triad = AgentTriad(*agents)

        bfi_results: Dict[str, Dict[str, int]] = {}
        n_q = len(BFI_QUESTIONS)
        for ag in agents:
            raw_scores = [
                ag.get_bfi_score(q, i + 1, n_q) for i, q in enumerate(BFI_QUESTIONS)
            ]
            bfi_results[ag.name] = compute_bfi_scores(raw_scores)

        # チームディレクトリに保存
        (logger.dir / "bfi_scores.json").write_text(
            json.dumps(bfi_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        dl = dataloader("mmlu", n_case=self.n_case)
        dl.set_mode("all")

        for idx in range(len(dl)):
            item = dl[idx]
            # --- after ---
            q_text, *opts = item["task_info"]          # タスク情報を展開
            question_text = (
                f"Question {idx + 1}:\n{q_text}\n"     # 見出し
                + "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(opts))
            )

            # --- Round 1 ---
            prompt_r1 = _TPL_R1.render(question_text=question_text)
            for ag in agents:
                ag.reset_history()

            triad.round_responses = {1: {}}
            for ag in agents:
                resp = ag.generate_response(prompt_r1, enforce_json=True)
                triad.round_responses[1][ag.name] = resp
                logger.log_turn(
                    {
                        "q_id": idx + 1,
                        "turn": 1,
                        "agent": ag.name,
                        **resp,
                        "tokens": count_tokens_tiktoken(
                            str(resp["reasoning"]) + str(resp["answer"])
                        ),
                    }
                )

            # # --- Round 2 & 3 ---
            # for turn in (2, 3):
            #     triad.round_responses[turn] = {}
            #     for ag in agents:
            #         other = [
            #             triad.round_responses[turn - 1][x.name]
            #             for x in agents
            #             if x.name != ag.name
            #         ]
            #         prompt_rn = _TPL_RN.render(
            #             turn=turn, other1=other[0], other2=other[1],personality=ag.personality_text
            #         )
            #         resp = ag.generate_response(prompt_rn, enforce_json=True)
            #         triad.round_responses[turn][ag.name] = resp
            #         logger.log_turn(
            #             {
            #                 "q_id": idx + 1,
            #                 "turn": turn,
            #                 "agent": ag.name,
            #                 **resp,
            #                 "tokens": count_tokens_tiktoken(
            #                     str(resp["reasoning"]) + str(resp["answer"])
            #                 ),
            #             }
            #         )

            # --- Consensus & Metrics ---
            final_ans = triad.get_final_consensus()
            total_tok = sum(
                count_tokens_tiktoken(str(r["reasoning"]) + str(r["answer"]))
                for round_dict in triad.round_responses.values()
                for r in round_dict.values()
            )
            logger.log_metric(
                {
                    "q_id": idx + 1,
                    "final_answer": final_ans,
                    "correct_answer": item["answer"],
                    "correct": final_ans == item["answer"].upper(),
                    "total_tokens": total_tok,
                }
            )

            # --- Run 終了時 Accuracy 集計 ---
        met = _pd.read_csv(logger.metrics_path)
        acc = met["correct"].mean()
        (logger.dir / "run_summary.json").write_text(
            json.dumps({"accuracy": acc,
                        "n_case": len(met),
                        "team": team_name}, indent=2),
            encoding="utf-8")