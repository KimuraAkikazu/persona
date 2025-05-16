import argparse
import random
import numpy as np
from llama_cpp import Llama
import config
from runner import ExperimentRunner
import yaml
import json
import hashlib

# ----- BigFive プロンプト辞書（±表記） -----
bigfive_prompts = {
    "++++-": (
        "You have high Openness, high Conscientiousness, high Extraversion, high Agreeableness, and low Neuroticism."
        "You are imaginative, organized, energetic, cooperative, and calm."
    ),
    "++-+-": (
        "You have high Openness, high Conscientiousness, low Extraversion, high Agreeableness, and low Neuroticism."
        "You are imaginative, organized, reserved, cooperative, and calm."
    ),
    "+++++": (
        "You have high Openness, high Conscientiousness, high Extraversion, high Agreeableness, and high Neuroticism."
        "You are imaginative, organized, energetic, cooperative, and emotionally intense."
    ),
    "----+" : (
        "You have low Openness, low Conscientiousness, low Extraversion, low Agreeableness, and high Neuroticism."
        "You tend to be conventional, disorganized, reserved, uncooperative, and anxious."
    ),
    "-----" : (
        "You have low Openness, low Conscientiousness, low Extraversion, low Agreeableness, and low Neuroticism."
        "You tend to be traditional, laid-back, quiet, independent, and stable."
    ),
    "--+-+" : (
        "You have low Openness, low Conscientiousness, high Extraversion, low Agreeableness, and high Neuroticism."
        "You tend to be practical, spontaneous, outgoing, critical, and nervous."
    ),
    "---++" : (
        "You have low Openness, low Conscientiousness, low Extraversion, high Agreeableness, and high Neuroticism."
        "You tend to be traditional, careless, introverted, friendly, and moody."
    ),
    "+++--" : (
        "You have high Openness, high Conscientiousness, high Extraversion, low Agreeableness, and low Neuroticism."
        "You tend to be curious, efficient, active, competitive, and relaxed."
    ),
    "NONE": "",
}

def pattern_to_prompt(pat: str) -> str:
    pat = pat.upper()
    return bigfive_prompts.get(pat, "") if pat != "NONE" else ""

def auto_team_name(members):
    uniq = set(members)
    if len(uniq) == 1:
        val = next(iter(uniq))
        return f"Team{val}" if val else "TeamNone"
    h = hashlib.md5(",".join(members).encode()).hexdigest()[:6]
    return f"TeamMixed_{h}"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--patterns",
        help="Comma-sep list of 3 patterns, e.g. '++---,++---,++---'"
    )
    p.add_argument(
        "--team_file",
        help="YAML/JSON file that defines teams and their members"
    )
    p.add_argument(
        "--n_case", type=int, default=50, help="Number of MMLU cases per run"
    )
    p.add_argument(
        "--n_runs", type=int, default=1, help="Repeat runs per team"
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # シード固定
    random.seed(args.seed)
    np.random.seed(args.seed)

    # モデルロード
    model_path = config.get_model_path()
    llama = Llama(
        model_path=model_path,
        chat_format="llama-3",
        n_ctx=8192,
        n_threads=8,
        n_gpu_layers=-1,
    )

    # チーム設定を動的生成
    team_map = {}
    if args.team_file:
        with open(args.team_file, "r", encoding="utf-8") as f:
            spec = yaml.safe_load(f) if args.team_file.endswith((".yml", ".yaml")) else json.load(f)
        for tname, conf in spec["teams"].items():
            members = [pattern_to_prompt(p) for p in conf["members"]]
            team_map[tname] = members
    elif args.patterns:
        pats = [p.strip() for p in args.patterns.split(",")]
        assert len(pats) == 3, "--patterns には 3 個指定してください"
        team_map[auto_team_name(pats)] = [pattern_to_prompt(p) for p in pats]
    else:
        team_map = {
            "TeamNone": ["", "", ""],
            "TeamMixed": [
                bigfive_prompts["++++-"],
                bigfive_prompts["++-+-"],
                bigfive_prompts["+++++"],
            ],
            "TeamPlusMinus": [bigfive_prompts["++-+-"]] * 3,
        }

    runner = ExperimentRunner(
        llama=llama,
        bigfive_prompts=bigfive_prompts,
        n_case=args.n_case,
        seed=args.seed,
        model_path=model_path,
    )

    for team_name, members in team_map.items():
        for _ in range(args.n_runs):
            runner.run_team(team_name, members)

if __name__ == "__main__":
    main()
