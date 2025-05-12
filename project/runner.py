# runner.py
from pathlib import Path
from typing import List, Dict, Any
from jinja2 import Template
from llama_cpp import Llama
from agents import LlamaAgent, AgentTriad
from dataloader import dataloader
from logger import ExperimentLogger
from utils import count_tokens_tiktoken


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
    ):
        self.llama = llama
        self.prompts = bigfive_prompts
        self.n_case = n_case
        self.seed = seed
        self.model_path = model_path

    def run_team(self, team_name: str, personalities: List[str]) -> None:
        logger = ExperimentLogger(
            team=team_name, seed=self.seed, model_path=self.model_path
        )

        agents = [
            LlamaAgent(f"Agent{i+1}", personalities[i], self.llama, max_tokens=1024)
            for i in range(3)
        ]
        triad = AgentTriad(*agents)

        dl = dataloader("mmlu", n_case=self.n_case)
        dl.set_mode("all")

        for idx in range(len(dl)):
            item = dl[idx]
            q_text = item["task_info"][0]
            opts = item["task_info"][1:]
            question_text = f"Question: {q_text}\n" + "\n".join(
                f"{chr(65+i)}. {opt}" for i, opt in enumerate(opts)
            )

            # --- Round 1 ---
            prompt_r1 = _TPL_R1.render(question_text=question_text)
            for ag in agents:
                ag.reset_history()

            triad.round_responses = {1: {}}
            for ag in agents:
                resp = ag.generate_response(prompt_r1)
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

            # --- Round 2 & 3 ---
            for turn in (2, 3):
                triad.round_responses[turn] = {}
                for ag in agents:
                    other = [
                        triad.round_responses[turn - 1][x.name]
                        for x in agents
                        if x.name != ag.name
                    ]
                    prompt_rn = _TPL_RN.render(
                        turn=turn, other1=other[0], other2=other[1]
                    )
                    resp = ag.generate_response(prompt_rn)
                    triad.round_responses[turn][ag.name] = resp
                    logger.log_turn(
                        {
                            "q_id": idx + 1,
                            "turn": turn,
                            "agent": ag.name,
                            **resp,
                            "tokens": count_tokens_tiktoken(
                                str(resp["reasoning"]) + str(resp["answer"])
                            ),
                        }
                    )

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
