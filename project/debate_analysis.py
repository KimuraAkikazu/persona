#!/usr/bin/env python
"""
Analyze cooperative dynamics in multi‑agent MMLU discussions
===========================================================

This script provides **two analysis flows** for debate logs produced by
`runner.py`:

1. **Linguistic feature analysis for Turns 2 & 3**
   * Sentiment polarity per agent (TextBlob)
   * Word‑cloud visualisations (overall, per agent)
2. **Answer‑transition statistics**
   * Majority (team‑level) transitions Correct→Wrong / Wrong→Correct
   * Per‑agent transitions Correct→Wrong / Wrong→Correct
   * Results are written to CSV.

Output is saved under `project/analysis/<run_id>/...`.  Directory layout
mirrors that of the `results` tree:  `<run_id>/<Team>/<runXX>/`.

Usage
-----
```bash
conda activate llama-test
python debate_analysis.py \
  --results-dir project/results \
  --run-id 20250519_153843 \
  --out-dir project/analysis
```
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS

# ---------------------- CLI --------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi‑agent cooperation analyzer")
    p.add_argument("--results-dir", default="project/results", help="Root of results tree")
    p.add_argument("--run-id", required=True, help="Run ID to analyse (folder name)")
    p.add_argument("--out-dir", default="project/analysis", help="Base directory to save outputs")
    p.add_argument(
        "--mmlu-path",
        default="project/eval_data/mmlu.pkl",
        help="Pickled dict mapping q_id -> correct option (A/B/C/D)",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ------------------- IO helpers ----------------- #

def load_debate(path: Path) -> pd.DataFrame:
    """Read debate.jsonl into a DataFrame (one row per utterance)."""
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    return pd.DataFrame(rows)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ---------------- Linguistic part --------------- #

def analyse_turn23_text(df: pd.DataFrame, out_dir: Path):
    """Generate sentiment bar‑plot and word‑clouds for turns 2 & 3."""
    # filter turns 2 and 3
    sub = df[df["turn"].isin([2, 3])].copy()
    if sub.empty:
        return
    # polarity per utterance
    sub["polarity"] = sub["reasoning"].apply(lambda t: TextBlob(str(t)).sentiment.polarity)
    # ---- bar plot per agent ----
    mean_pol = sub.groupby("agent")["polarity"].mean().reset_index()
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=mean_pol, x="agent", y="polarity")
    ax.set_title("平均感情極性 (Turns 2 & 3)")
    ax.set_ylabel("polarity")
    plt.tight_layout()
    plt.savefig(out_dir / "avg_polarity_turn23.png", dpi=300)
    plt.close()

    # ---- word‑clouds ----
    all_text = "\n".join(sub["reasoning"].fillna("") + " " + sub["answer"].fillna(""))
    _save_wordcloud(all_text, "WordCloud 全体 (T2&3)", out_dir / "wordcloud_turn23_all.png")

    for agent, ag_df in sub.groupby("agent"):
        text = "\n".join(ag_df["reasoning"].fillna("") + " " + ag_df["answer"].fillna(""))
        _save_wordcloud(text, f"{agent} WordCloud (T2&3)", out_dir / f"wordcloud_{agent}_t23.png")


def _save_wordcloud(text: str, title: str, out_path: Path):
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=set(STOPWORDS),
        collocations=False,
    ).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ------------- Transition‑stat helpers ---------- #

def majority_vote(answers: list[str]) -> str | None:
    if not answers:
        return None
    cnt = Counter(answers)
    # in case of tie, deterministic (alphabetical) choice
    return sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[0][0]


def calc_transitions(
    df: pd.DataFrame, correct_map: dict[int, str]
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Return per‑agent and majority transition counts.

    *per_agent*  : {agent: {"CW": n, "WC": m}}
    *majority*   : {"CW": n, "WC": m}
    """
    per_agent = defaultdict(lambda: {"CW": 0, "WC": 0})
    majority = {"CW": 0, "WC": 0}

    # group by question id
    for q_id, q_df in df.groupby("q_id"):
        correct = correct_map.get(q_id)
        if correct is None:
            continue
        # sort turns
        turns = sorted(q_df["turn"].unique())
        # per‑agent answers keyed by turn
        answers_by_agent_turn: dict[str, dict[int, str]] = defaultdict(dict)
        for _idx, row in q_df.iterrows():
            answers_by_agent_turn[row["agent"]][row["turn"]] = row["answer"]
        # majority answers per turn
        maj_by_turn = {}
        for t in turns:
            maj_by_turn[t] = majority_vote(q_df[q_df["turn"] == t]["answer"].tolist())
        # ----- compute transitions (turn t -> t+1) -----
        for t_prev, t_next in zip(turns[:-1], turns[1:]):
            # majority
            prev_m = maj_by_turn[t_prev]
            next_m = maj_by_turn[t_next]
            if prev_m == correct and next_m != correct:
                majority["CW"] += 1
            elif prev_m != correct and next_m == correct:
                majority["WC"] += 1
            # per‑agent
            for ag, t_dict in answers_by_agent_turn.items():
                if t_prev not in t_dict or t_next not in t_dict:
                    continue
                prev_a, next_a = t_dict[t_prev], t_dict[t_next]
                if prev_a == correct and next_a != correct:
                    per_agent[ag]["CW"] += 1
                elif prev_a != correct and next_a == correct:
                    per_agent[ag]["WC"] += 1
    return per_agent, majority


def save_transition_csv(
    per_agent: dict[str, dict[str, int]],
    majority: dict[str, int],
    out_dir: Path,
):
    # majority summary
    pd.DataFrame([majority]).to_csv(out_dir / "majority_transition_summary.csv", index=False)
    # per‑agent summary
    rows = [
        {"agent": ag, "correct_to_wrong": v["CW"], "wrong_to_correct": v["WC"]}
        for ag, v in per_agent.items()
    ]
    pd.DataFrame(rows).to_csv(out_dir / "agent_transition_summary.csv", index=False)


# -------------------- Main loop ----------------- #

def analyse_run(run_dir: Path, correct_map: dict[int, str], out_root: Path, verbose=False):
    """Analyse a single run directory containing debate.jsonl"""
    debate_path = run_dir / "debate.jsonl"
    if not debate_path.exists():
        if verbose:
            print(f"  [skip] no debate.jsonl in {run_dir.relative_to(run_dir.parent.parent)}")
        return
    df = load_debate(debate_path)
    # output dir mirrors results tree
    rel = run_dir.relative_to(run_dir.parents[2])  # get Team/runXX relative path
    out_dir = out_root / rel
    ensure_dir(out_dir)

    # ---- 1) linguistic features ----
    analyse_turn23_text(df, out_dir)

    # ---- 2 & 3) transition statistics ----
    per_agent, majority = calc_transitions(df, correct_map)
    save_transition_csv(per_agent, majority, out_dir)

    if verbose:
        print(f"  ✔ analysed {rel}")


def main():
    args = parse_args()
    results_root = Path(args.results_dir) / args.run_id
    
    # Check if the results directory exists
    if not results_root.exists():
        print(f"Error: Results directory {results_root} does not exist.")
        return 1
    
    out_root = Path(args.out_dir) / args.run_id
    ensure_dir(out_root)

    # Check if the MMLU path exists
    mmlu_path = Path(args.mmlu_path)
    if not mmlu_path.exists():
        print(f"Error: MMLU data file {mmlu_path} does not exist.")
        return 1

    # load correct answers dict
    correct_map: dict[int, str]
    try:
        with open(mmlu_path, "rb") as f:
            correct_map = pickle.load(f)
    except Exception as e:
        print(f"Error loading MMLU data: {e}")
        return 1

    # Print summary info
    print(f"Analysis for run: {args.run_id}")
    print(f"Results directory: {results_root}")
    print(f"Output directory: {out_root}")
    print(f"MMLU data: {mmlu_path}")

    # Check run directory structure - try to find run subdirectories
    runs = list(results_root.glob("run*"))
    # If we found run directories directly under results_root, we're analyzing run format
    if runs:
        print(f"Found {len(runs)} run directories directly under results root")
        for run_dir in sorted(runs):
            analyse_run(run_dir, correct_map, out_root, verbose=args.verbose)
    else:
        # If no run directories found directly, traverse teams/runXX
        team_dirs = [p for p in results_root.iterdir() if p.is_dir()]
        if not team_dirs:
            print(f"No team directories found in {results_root}")
            return 1
            
        print(f"Found {len(team_dirs)} team directories")
        for team_dir in team_dirs:
            run_dirs = sorted([p for p in team_dir.iterdir() if p.is_dir()])
            print(f"  {team_dir.name}: {len(run_dirs)} run directories")
            for run_dir in run_dirs:
                analyse_run(run_dir, correct_map, out_root, verbose=args.verbose)

    print("Finished debate_analysis.py ✔")
    return 0


if __name__ == "__main__":
    exit(main())