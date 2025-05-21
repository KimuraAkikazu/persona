#!/usr/bin/env python
# custom_mmlu_analyzer.py

import argparse
import json
from pathlib import Path
import pandas as pd
from collections import Counter
import pickle
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os

# デフォルトのMMLUデータパス (スクリプトの場所からの相対パスを想定)
# 実行時のカレントディレクトリが 'project' の親である場合や、
# results_dir から遡って project ルートを見つける方がより堅牢です。
# ここでは results_dir の親を project ルートと仮定します。
DEFAULT_MMLU_PKL_NAME = "mmlu.pkl"
DEFAULT_EVAL_DATA_SUBDIR = "eval_data"

# ログファイル名
DEBATE_LOG_FILENAME = "debate.jsonl"

# 分析結果を格納する基本フォルダ名
ANALYSIS_BASE_DIR_NAME = "analysis"
PROJECT_DIR_NAME = "project" # 'project' ディレクトリの一般的な名前

# 出力サブディレクトリ
LINGUISTIC_SUBDIR = "linguistic_analysis"
ACCURACY_SUBDIR = "accuracy_analysis"


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Agent MMLU Debate Log Analyzer")
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to the results directory (e.g., project/results)",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="The specific run_id to analyze (e.g., 20250519_153843)",
    )
    # 出力ディレクトリは project/analysis/<run_id> で固定とするため、引数からは削除
    # parser.add_argument("--out-dir", default="project/analysis", help="Base directory for analysis output")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def load_mmlu_answers(project_root_dir: Path, verbose: bool = False) -> dict:
    """
    Loads MMLU answers from mmlu.pkl.
    Assumes mmlu.pkl is a dictionary mapping q_id (int) to correct answer (str: 'A'/'B'/'C'/'D').
    """
    mmlu_pkl_path = project_root_dir / DEFAULT_EVAL_DATA_SUBDIR / DEFAULT_MMLU_PKL_NAME
    dummy_answers = {i: "A" for i in range(1, 101)}  # Default 100 dummy questions

    if not mmlu_pkl_path.exists():
        if verbose:
            print(
                f"WARNING: MMLU data file not found at {mmlu_pkl_path}. Using dummy data."
            )
        return dummy_answers

    try:
        with open(mmlu_pkl_path, "rb") as f:
            mmlu_data = pickle.load(f)
        
        # Validate format (dict with int keys and str values A,B,C,D)
        if isinstance(mmlu_data, dict) and \
           all(isinstance(k, int) for k in mmlu_data.keys()) and \
           all(isinstance(v, str) and v in ['A', 'B', 'C', 'D', 'E'] for v in mmlu_data.values()): # MMLU can have E
            if verbose:
                print(f"Successfully loaded MMLU answers from {mmlu_pkl_path}.")
            return mmlu_data
        else:
            if verbose:
                print(
                    f"WARNING: MMLU data at {mmlu_pkl_path} is not in the expected format "
                    "(dict: q_id (int) -> answer (str 'A'-'E')). Using dummy data."
                )
            return dummy_answers
    except Exception as e:
        if verbose:
            print(
                f"ERROR: Failed to load or parse MMLU data from {mmlu_pkl_path}: {e}. Using dummy data."
            )
        return dummy_answers


def load_debate_jsonl(file_path: Path) -> pd.DataFrame:
    """Loads a single debate.jsonl file into a pandas DataFrame."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line in {file_path}: {line.strip()}")
    return pd.DataFrame(records)


def generate_wordcloud_image(text: str, title: str, out_path: Path):
    """Generates and saves a word cloud image."""
    if not text.strip():
        print(f"Skipping word cloud for '{title}' due to empty text.")
        return
    try:
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
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating word cloud for '{title}': {e}")


def analyze_linguistic_features(df_run: pd.DataFrame, output_dir: Path, team_name: str, run_name: str):
    """
    Analyzes linguistic features for turns 2 and 3.
    - Sentiment analysis of 'reasoning'.
    - Word cloud of 'reasoning'.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter for turns 2 and 3
    df_turns_2_3 = df_run[df_run["turn"].isin([2, 3])].copy()
    if df_turns_2_3.empty:
        print(f"  No data for turns 2 or 3 in {team_name}/{run_name} for linguistic analysis.")
        return

    # Sentiment Analysis
    df_turns_2_3["reasoning_polarity"] = df_turns_2_3["reasoning"].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0.0
    )
    
    sentiment_summary = (
        df_turns_2_3.groupby(["turn", "agent"])["reasoning_polarity"]
        .mean()
        .reset_index()
    )
    sentiment_overall_turn = (
        df_turns_2_3.groupby("turn")["reasoning_polarity"].mean().reset_index()
    )
    sentiment_overall_turn["agent"] = "Overall"
    
    final_sentiment_summary = pd.concat([sentiment_summary, sentiment_overall_turn], ignore_index=True)
    sentiment_csv_path = output_dir / "sentiment_reasoning_turns_2_3.csv"
    final_sentiment_summary.to_csv(sentiment_csv_path, index=False)
    print(f"    Linguistic: Saved sentiment analysis to {sentiment_csv_path}")

    # Word Clouds
    for turn_num in [2, 3]:
        turn_text = " ".join(
            df_turns_2_3[df_turns_2_3["turn"] == turn_num]["reasoning"].fillna("")
        )
        wc_title = f"{team_name}/{run_name} - Turn {turn_num} Reasoning"
        wc_path = output_dir / f"wordcloud_turn{turn_num}_reasoning.png"
        generate_wordcloud_image(turn_text, wc_title, wc_path)
        if turn_text.strip():
             print(f"    Linguistic: Generated word cloud for Turn {turn_num} to {wc_path}")


def get_team_answer(answers: list) -> str:
    """Determines team answer by majority vote. Handles ties by returning 'N/A'."""
    if not answers:
        return "N/A"
    counts = Counter(answers)
    max_count = 0
    candidates = []
    for item, count in counts.items():
        if count > max_count:
            max_count = count
            candidates = [item]
        elif count == max_count:
            candidates.append(item)
    
    if len(candidates) == 1:
        return candidates[0]
    else: # Tie or no clear majority (e.g. if answers can be other than A/B/C/D)
        # For 3 agents, if all provide A/B/C/D, a tie is (A,B,C) - this logic means N/A
        # If we want to force a choice, sort candidates and pick first.
        # For now, N/A on ties to be conservative.
        return "N/A"


def analyze_team_accuracy_transitions(df_run: pd.DataFrame, mmlu_answers: dict, output_dir: Path):
    """
    Analyzes team-level accuracy transitions (correct/incorrect) across turns.
    Saves a CSV with counts of transitions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    transitions = {
        "t1_correct_to_t2_incorrect": 0,
        "t1_incorrect_to_t2_correct": 0,
        "t2_correct_to_t3_incorrect": 0,
        "t2_incorrect_to_t3_correct": 0,
    }

    if "q_id" not in df_run.columns:
        print("    Accuracy: 'q_id' column missing. Skipping team accuracy transitions.")
        return
        
    for q_id, group in df_run.groupby("q_id"):
        correct_answer = mmlu_answers.get(q_id)
        if correct_answer is None:
            # print(f"    Warning: No MMLU answer found for q_id {q_id}. Skipping this question for accuracy.")
            continue

        turn_answers_is_correct = {} # Store T/F for turn 1, 2, 3

        for turn_num in [1, 2, 3]:
            turn_data = group[group["turn"] == turn_num]
            if turn_data.empty:
                turn_answers_is_correct[turn_num] = None # No data for this turn
                continue
            
            agent_answers_for_turn = turn_data["answer"].tolist()
            team_ans = get_team_answer(agent_answers_for_turn)
            
            if team_ans == "N/A":
                 turn_answers_is_correct[turn_num] = None # Undecided
            else:
                turn_answers_is_correct[turn_num] = (team_ans == correct_answer)

        # Evaluate transitions
        # T1 -> T2
        if turn_answers_is_correct.get(1) is True and turn_answers_is_correct.get(2) is False:
            transitions["t1_correct_to_t2_incorrect"] += 1
        if turn_answers_is_correct.get(1) is False and turn_answers_is_correct.get(2) is True:
            transitions["t1_incorrect_to_t2_correct"] += 1
        
        # T2 -> T3
        if turn_answers_is_correct.get(2) is True and turn_answers_is_correct.get(3) is False:
            transitions["t2_correct_to_t3_incorrect"] += 1
        if turn_answers_is_correct.get(2) is False and turn_answers_is_correct.get(3) is True:
            transitions["t2_incorrect_to_t3_correct"] += 1
            
    summary_df = pd.DataFrame(list(transitions.items()), columns=["transition_type", "count"])
    csv_path = output_dir / "team_accuracy_transitions.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"    Accuracy: Saved team accuracy transitions to {csv_path}")


def analyze_agent_accuracy_changes(df_run: pd.DataFrame, mmlu_answers: dict, output_dir: Path):
    """
    Analyzes individual agent accuracy changes from T1->T2 and T2->T3.
    Saves a CSV with counts of changes per agent.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    agent_changes_list = []

    if "q_id" not in df_run.columns or "agent" not in df_run.columns:
        print("    Accuracy: 'q_id' or 'agent' column missing. Skipping agent accuracy changes.")
        return

    agents = df_run["agent"].unique()
    for agent_name in agents:
        agent_transitions = {
            "t1_correct_to_t2_incorrect": 0, "t1_incorrect_to_t2_correct": 0,
            "t2_correct_to_t3_incorrect": 0, "t2_incorrect_to_t3_correct": 0,
        }
        agent_df = df_run[df_run["agent"] == agent_name]

        for q_id, group in agent_df.groupby("q_id"):
            correct_answer = mmlu_answers.get(q_id)
            if correct_answer is None:
                continue

            ans_t1 = group[group["turn"] == 1]["answer"].iloc[0] if not group[group["turn"] == 1].empty else None
            ans_t2 = group[group["turn"] == 2]["answer"].iloc[0] if not group[group["turn"] == 2].empty else None
            ans_t3 = group[group["turn"] == 3]["answer"].iloc[0] if not group[group["turn"] == 3].empty else None

            is_correct_t1 = (ans_t1 == correct_answer) if ans_t1 is not None else None
            is_correct_t2 = (ans_t2 == correct_answer) if ans_t2 is not None else None
            is_correct_t3 = (ans_t3 == correct_answer) if ans_t3 is not None else None
            
            # T1 -> T2 (Turn 2 is "2ターン目以降")
            if is_correct_t1 is True and is_correct_t2 is False:
                agent_transitions["t1_correct_to_t2_incorrect"] += 1
            if is_correct_t1 is False and is_correct_t2 is True:
                agent_transitions["t1_incorrect_to_t2_correct"] += 1
            
            # T2 -> T3 (Turn 3 is "2ターン目以降")
            if is_correct_t2 is True and is_correct_t3 is False:
                agent_transitions["t2_correct_to_t3_incorrect"] += 1
            if is_correct_t2 is False and is_correct_t3 is True:
                agent_transitions["t2_incorrect_to_t3_correct"] += 1
        
        for change_type, count in agent_transitions.items():
            agent_changes_list.append({"agent": agent_name, "change_type": change_type, "count": count})

    summary_df = pd.DataFrame(agent_changes_list)
    csv_path = output_dir / "agent_accuracy_changes.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"    Accuracy: Saved agent accuracy changes to {csv_path}")


def main():
    args = parse_args()
    
    results_base_dir = Path(args.results_dir)
    # The project root is assumed to be the parent of results_dir
    project_root_dir = results_base_dir.parent 
    
    # Define the main analysis output directory for the run_id
    # project/analysis/<run_id>
    main_analysis_dir_for_run_id = project_root_dir / ANALYSIS_BASE_DIR_NAME / args.run_id
    
    if args.verbose:
        print(f"Starting analysis for run_id: {args.run_id}")
        print(f"Results directory: {results_base_dir}")
        print(f"Project root (assumed): {project_root_dir}")
        print(f"Base analysis output directory: {main_analysis_dir_for_run_id}")

    mmlu_answers = load_mmlu_answers(project_root_dir, args.verbose)

    run_id_path = results_base_dir / args.run_id
    if not run_id_path.exists() or not run_id_path.is_dir():
        print(f"ERROR: Run ID directory not found: {run_id_path}")
        return

    # Iterate through teams
    for team_dir in run_id_path.iterdir():
        if not team_dir.is_dir():
            continue
        team_name = team_dir.name
        if args.verbose:
            print(f"\nProcessing Team: {team_name}")

        # Iterate through run directories (run00, run01, etc.)
        for run_subdir in team_dir.iterdir():
            if not run_subdir.is_dir() or not run_subdir.name.startswith("run"):
                continue
            run_name = run_subdir.name
            debate_file_path = run_subdir / DEBATE_LOG_FILENAME
            
            if not debate_file_path.exists():
                if args.verbose:
                    print(f"  Skipping {team_name}/{run_name}: {DEBATE_LOG_FILENAME} not found.")
                continue

            if args.verbose:
                print(f"  Analyzing {team_name}/{run_name}/{DEBATE_LOG_FILENAME}")
            
            df_run_log = load_debate_jsonl(debate_file_path)
            if df_run_log.empty:
                if args.verbose:
                    print(f"  Skipping {team_name}/{run_name}: Log file is empty or invalid.")
                continue

            # Define specific output directory for this run's analysis
            # project/analysis/<run_id>/<TeamName>/<run_dir>/
            output_dir_for_this_run = main_analysis_dir_for_run_id / team_name / run_name
            
            # 1. Linguistic Features Analysis
            linguistic_output_dir = output_dir_for_this_run / LINGUISTIC_SUBDIR
            analyze_linguistic_features(df_run_log, linguistic_output_dir, team_name, run_name)

            # 2. Team Accuracy Transitions
            accuracy_output_dir = output_dir_for_this_run / ACCURACY_SUBDIR
            analyze_team_accuracy_transitions(df_run_log, mmlu_answers, accuracy_output_dir)
            
            # 3. Agent Accuracy Changes
            analyze_agent_accuracy_changes(df_run_log, mmlu_answers, accuracy_output_dir) # Uses same accuracy_output_dir

    print("\nAnalysis complete.")
    print(f"Results saved under: {main_analysis_dir_for_run_id}")


if __name__ == "__main__":
    # Ensure matplotlib does not try to use GUI backend if not available
    plt.switch_backend('Agg') 
    main()