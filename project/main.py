# main.py
import argparse
import random
import numpy as np
from llama_cpp import Llama
import config
from runner import ExperimentRunner

# ----- CLI 引数 -----
p = argparse.ArgumentParser()
p.add_argument(
    "--teams",
    default="TeamNone,TeamMixed,TeamT2",
    help="Comma-separated list of team names to run",
)
p.add_argument(
    "--n_case", type=int, default=50, help="Number of MMLU cases to run per team"
)
p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = p.parse_args()

# ----- シード固定 -----
random.seed(args.seed)
np.random.seed(args.seed)

# ----- モデルロード -----
model_path = config.get_model_path()
llama = Llama(
    model_path=model_path,
    chat_format="llama-3",
    n_ctx=8192,
    n_threads=8,
    n_gpu_layers=-1,
)

# ----- BigFive プロンプト辞書 -----
bigfive_prompts = {
    "AgentT1": (
        "You are a character with high Openness and high Agreeableness, "
        "paired with moderate Extraversion and moderate Conscientiousness. "
        "Your imaginative mind and warm, cooperative nature drive you to explore innovative ideas "
        "while nurturing harmonious interactions. "
        "You remain calm (low Neuroticism) even when facing challenges. "
        "Answer thoughtfully and creatively, ensuring your responses reflect empathy and originality."
    ),
    "AgentT2": (
        "You are a character with high Conscientiousness and high Extraversion, "
        "complemented by moderate Openness and low Agreeableness. "
        "Your decisive, organized, and assertive demeanor makes you a pragmatic leader who values efficiency and clarity. "
        "You maintain composure (low Neuroticism) and focus on delivering clear, goal-oriented responses without excessive sentiment. "
        "Answer in a direct and methodical manner, staying true to your results-driven mindset."
    ),
    "AgentT3": (
        "You are a character with high Openness and high Neuroticism, "
        "along with moderate levels of Conscientiousness, Extraversion, and Agreeableness. "
        "Your rich inner life fuels a deep creative insight, though it is often accompanied by intense emotional sensitivity "
        "and occasional self-doubt. "
        "Embrace your introspective and passionate nature; answer with nuanced, reflective responses that capture both your "
        "visionary ideas and your candid vulnerability."
    ),
    "AgentNone": "",
}

# ----- チーム設定 -----
team_map = {
    "TeamNone": ["", "", ""],
    "TeamMixed": [
        bigfive_prompts["AgentT1"],
        bigfive_prompts["AgentT2"],
        bigfive_prompts["AgentT3"],
    ],
    "TeamT2": [
        bigfive_prompts["AgentT2"],
        bigfive_prompts["AgentT2"],
        bigfive_prompts["AgentT2"],
    ],
}

# ----- 実行 -----
runner = ExperimentRunner(
    llama=llama,
    bigfive_prompts=bigfive_prompts,
    n_case=args.n_case,
    seed=args.seed,
    model_path=model_path,
)

for team in args.teams.split(","):
    if team not in team_map:
        print(f"Warning: unknown team '{team}', skipping")
        continue
    runner.run_team(team, team_map[team])

print("All experiments finished. Check results/<run_id>/<team> directories.")
