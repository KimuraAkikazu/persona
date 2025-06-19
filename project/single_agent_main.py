#!/usr/bin/env python3
# project/single_agent_main.py
"""
Single-agent MMLU runner (1-turn only) - Modified for statistical validation

指定された性格特性リストに対し、MMLUタスクを指定回数繰り返し実行し、
性格ごとの平均正答率と標準偏差を算出するスクリプト。

使い方例
────────
conda activate llama-test
python single_agent_main.py
python single_agent_main.py --seed 42  # 乱数シードのベース値を変更
"""

from __future__ import annotations
import argparse, datetime, json, random, csv
from pathlib import Path

import numpy as np
from llama_cpp import Llama

import config
from agents import LlamaAgent
from dataloader import dataloader
from logger import ExperimentLogger


# ----------------------------------------------------------------------
# Big-Five プロンプト辞書（変更なし）
# ----------------------------------------------------------------------
BIGFIVE: dict[str, str] = {
    "+++++": "1. Extraversion: Warm up quickly to others. Show my feelings when I’m happy. 2. Conscientiousness: Get things done quickly. Carry out my plans. Like order. Keep things tidy. 3. Agreeableness: Feel others’ emotions. Inquire about others’ well-being. Respect authority. 4. Openness: Am quick to understand things. Like to solve complex problems. 5. Neuroticism: Get angry easily. Get upset easily. Can be stirred up easily.",
    "++++-": "1. Extraversion: Warm up quickly to others. Show my feelings when I’m happy. 2. Conscientiousness: Get things done quickly. Carry out my plans. Like order. Keep things tidy. 3. Agreeableness: Feel others’ emotions. Inquire about others’ well-being. Respect authority. 4. Openness: Am quick to understand things. Like to solve complex problems. 5. Neuroticism: Rarely get irritated. Keep my emotions under control. Seldom feel blue.",
    "+++-+": "1. Extraversion: Warm up quickly to others. Show my feelings when I’m happy. 2. Conscientiousness: Get things done quickly. Carry out my plans. Like order. Keep things tidy. 3. Agreeableness: Feel others’ emotions. Inquire about others’ well-being. Respect authority. 4. Openness: Have difficulty understanding abstract ideas. Seldom get lost in thought. 5. Neuroticism: Get angry easily. Get upset easily. Can be stirred up easily.",
    "+++--": "1. Extraversion: Warm up quickly to others. Show my feelings when I’m happy. 2. Conscientiousness: Get things done quickly. Carry out my plans. Like order. Keep things tidy. 3. Agreeableness: Feel others’ emotions. Inquire about others’ well-being. Respect authority. 4. Openness: Have difficulty understanding abstract ideas. Seldom get lost in thought. 5. Neuroticism: Rarely get irritated. Keep my emotions under control. Seldom feel blue.",
    "++-++": "1. Extraversion: Warm up quickly to others. Show my feelings when I’m happy. 2. Conscientiousness: Get things done quickly. Carry out my plans. Like order. Keep things tidy. 3. Agreeableness: Am not interested in other people’s problems. Insult people. 4. Openness: Am quick to understand things. Like to solve complex problems. 5. Neuroticism: Get angry easily. Get upset easily. Can be stirred up easily.",
    "++-+-": "1. Extraversion: Warm up quickly to others. Show my feelings when I’m happy. 2. Conscientiousness: Get things done quickly. Carry out my plans. Like order. Keep things tidy. 3. Agreeableness: Am not interested in other people’s problems. Insult people. 4. Openness: Am quick to understand things. Like to solve complex problems. 5. Neuroticism: Rarely get irritated. Keep my emotions under control. Seldom feel blue.",
    "++--+": "1. Extraversion: Warm up quickly to others. Show my feelings when I’m happy. 2. Conscientiousness: Get things done quickly. Carry out my plans. Like order. Keep things tidy. 3. Agreeableness: Am not interested in other people’s problems. Insult people. 4. Openness: Have difficulty understanding abstract ideas. Seldom get lost in thought. 5. Neuroticism: Get angry easily. Get upset easily. Can be stirred up easily.",
    "++---": "1. Extraversion: Warm up quickly to others. Show my feelings when I’m happy. 2. Conscientiousness: Get things done quickly. Carry out my plans. Like order. Keep things tidy. 3. Agreeableness: Am not interested in other people’s problems. Insult people. 4. Openness: Have difficulty understanding abstract ideas. Seldom get lost in thought. 5. Neuroticism: Rarely get irritated. Keep my emotions under control. Seldom feel blue.",
    "+-+++": "1. Extraversion: Warm up quickly to others. Show my feelings when I’m happy. 2. Conscientiousness: Waste my time. Leave my belongings around. 3. Agreeableness: Feel others’ emotions. Inquire about others’ well-being. Respect authority. 4. Openness: Am quick to understand things. Like to solve complex problems. 5. Neuroticism: Get angry easily. Get upset easily. Can be stirred up easily.",
    "+-++-": "1. Extraversion: Warm up quickly to others. Show my feelings when I’m happy. 2. Conscientiousness: Waste my time. Leave my belongings around. 3. Agreeableness: Feel others’ emotions. Inquire about others’ well-being. Respect authority. 4. Openness: Am quick to understand things. Like to solve complex problems. 5. Neuroticism: Rarely get irritated. Keep my emotions under control. Seldom feel blue.",
    "+-+-+": "1. Extraversion: Warm up quickly to others. Show my feelings when I’m happy. 2. Conscientiousness: Waste my time. Leave my belongings around. 3. Agreeableness: Feel others’ emotions. Inquire about others’ well-being. Respect authority. 4. Openness: Have difficulty understanding abstract ideas. Seldom get lost in thought. 5. Neuroticism: Get angry easily. Get upset easily. Can be stirred up easily.",
    "+-+--": "1. Extraversion: Warm up quickly to others. Show my feelings when I’m happy. 2. Conscientiousness: Waste my time. Leave my belongings around. 3. Agreeableness: Feel others’ emotions. Inquire about others’ well-being. Respect authority. 4. Openness: Have difficulty understanding abstract ideas. Seldom get lost in thought. 5. Neuroticism: Rarely get irritated. Keep my emotions under control. Seldom feel blue.",
    "+--++": "1. Extraversion: Warm up quickly to others. Show my feelings when I’m happy. 2. Conscientiousness: Waste my time. Leave my belongings around. 3. Agreeableness: Am not interested in other people’s problems. Insult people. 4. Openness: Am quick to understand things. Like to solve complex problems. 5. Neuroticism: Get angry easily. Get upset easily. Can be stirred up easily.",
    "+--+-": "1. Extraversion: Warm up quickly to others. Show my feelings when I’m happy. 2. Conscientiousness: Waste my time. Leave my belongings around. 3. Agreeableness: Am not interested in other people’s problems. Insult people. 4. Openness: Am quick to understand things. Like to solve complex problems. 5. Neuroticism: Rarely get irritated. Keep my emotions under control. Seldom feel blue.",
    "+---+": "1. Extraversion: Warm up quickly to others. Show my feelings when I’m happy. 2. Conscientiousness: Waste my time. Leave my belongings around. 3. Agreeableness: Am not interested in other people’s problems. Insult people. 4. Openness: Have difficulty understanding abstract ideas. Seldom get lost in thought. 5. Neuroticism: Get angry easily. Get upset easily. Can be stirred up easily.",
    "+----": "1. Extraversion: Warm up quickly to others. Show my feelings when I’m happy. 2. Conscientiousness: Waste my time. Leave my belongings around. 3. Agreeableness: Am not interested in other people’s problems. Insult people. 4. Openness: Have difficulty understanding abstract ideas. Seldom get lost in thought. 5. Neuroticism: Rarely get irritated. Keep my emotions under control. Seldom feel blue.",
    "-++++": "1. Extraversion: Am hard to get to know. Keep others at a distance. 2. Conscientiousness: Get things done quickly. Carry out my plans. Like order. Keep things tidy. 3. Agreeableness: Feel others’ emotions. Inquire about others’ well-being. Respect authority. 4. Openness: Am quick to understand things. Like to solve complex problems. 5. Neuroticism: Get angry easily. Get upset easily. Can be stirred up easily.",
    "-+++-": "1. Extraversion: Am hard to get to know. Keep others at a distance. 2. Conscientiousness: Get things done quickly. Carry out my plans. Like order. Keep things tidy. 3. Agreeableness: Feel others’ emotions. Inquire about others’ well-being. Respect authority. 4. Openness: Am quick to understand things. Like to solve complex problems. 5. Neuroticism: Rarely get irritated. Keep my emotions under control. Seldom feel blue.",
    "-++-+": "1. Extraversion: Am hard to get to know. Keep others at a distance. 2. Conscientiousness: Get things done quickly. Carry out my plans. Like order. Keep things tidy. 3. Agreeableness: Feel others’ emotions. Inquire about others’ well-being. Respect authority. 4. Openness: Have difficulty understanding abstract ideas. Seldom get lost in thought. 5. Neuroticism: Get angry easily. Get upset easily. Can be stirred up easily.",
    "-++--": "1. Extraversion: Am hard to get to know. Keep others at a distance. 2. Conscientiousness: Get things done quickly. Carry out my plans. Like order. Keep things tidy. 3. Agreeableness: Feel others’ emotions. Inquire about others’ well-being. Respect authority. 4. Openness: Have difficulty understanding abstract ideas. Seldom get lost in thought. 5. Neuroticism: Rarely get irritated. Keep my emotions under control. Seldom feel blue.",
    "-+-++": "1. Extraversion: Am hard to get to know. Keep others at a distance. 2. Conscientiousness: Get things done quickly. Carry out my plans. Like order. Keep things tidy. 3. Agreeableness: Am not interested in other people’s problems. Insult people. 4. Openness: Am quick to understand things. Like to solve complex problems. 5. Neuroticism: Get angry easily. Get upset easily. Can be stirred up easily.",
    "-+-+-": "1. Extraversion: Am hard to get to know. Keep others at a distance. 2. Conscientiousness: Get things done quickly. Carry out my plans. Like order. Keep things tidy. 3. Agreeableness: Am not interested in other people’s problems. Insult people. 4. Openness: Am quick to understand things. Like to solve complex problems. 5. Neuroticism: Rarely get irritated. Keep my emotions under control. Seldom feel blue.",
    "-+--+": "1. Extraversion: Am hard to get to know. Keep others at a distance. 2. Conscientiousness: Get things done quickly. Carry out my plans. Like order. Keep things tidy. 3. Agreeableness: Am not interested in other people’s problems. Insult people. 4. Openness: Have difficulty understanding abstract ideas. Seldom get lost in thought. 5. Neuroticism: Get angry easily. Get upset easily. Can be stirred up easily.",
    "-+---": "1. Extraversion: Am hard to get to know. Keep others at a distance. 2. Conscientiousness: Get things done quickly. Carry out my plans. Like order. Keep things tidy. 3. Agreeableness: Am not interested in other people’s problems. Insult people. 4. Openness: Have difficulty understanding abstract ideas. Seldom get lost in thought. 5. Neuroticism: Rarely get irritated. Keep my emotions under control. Seldom feel blue.",
    "--+++": "1. Extraversion: Am hard to get to know. Keep others at a distance. 2. Conscientiousness: Waste my time. Leave my belongings around. 3. Agreeableness: Feel others’ emotions. Inquire about others’ well-being. Respect authority. 4. Openness: Am quick to understand things. Like to solve complex problems. 5. Neuroticism: Get angry easily. Get upset easily. Can be stirred up easily.",
    "--++-": "1. Extraversion: Am hard to get to know. Keep others at a distance. 2. Conscientiousness: Waste my time. Leave my belongings around. 3. Agreeableness: Feel others’ emotions. Inquire about others’ well-being. Respect authority. 4. Openness: Am quick to understand things. Like to solve complex problems. 5. Neuroticism: Rarely get irritated. Keep my emotions under control. Seldom feel blue.",
    "--+-+": "1. Extraversion: Am hard to get to know. Keep others at a distance. 2. Conscientiousness: Waste my time. Leave my belongings around. 3. Agreeableness: Feel others’ emotions. Inquire about others’ well-being. Respect authority. 4. Openness: Have difficulty understanding abstract ideas. Seldom get lost in thought. 5. Neuroticism: Get angry easily. Get upset easily. Can be stirred up easily.",
    "--+--": "1. Extraversion: Am hard to get to know. Keep others at a distance. 2. Conscientiousness: Waste my time. Leave my belongings around. 3. Agreeableness: Feel others’ emotions. Inquire about others’ well-being. Respect authority. 4. Openness: Have difficulty understanding abstract ideas. Seldom get lost in thought. 5. Neuroticism: Rarely get irritated. Keep my emotions under control. Seldom feel blue.",
    "---++": "1. Extraversion: Am hard to get to know. Keep others at a distance. 2. Conscientiousness: Waste my time. Leave my belongings around. 3. Agreeableness: Am not interested in other people’s problems. Insult people. 4. Openness: Am quick to understand things. Like to solve complex problems. 5. Neuroticism: Get angry easily. Get upset easily. Can be stirred up easily.",
    "---+-": "1. Extraversion: Am hard to get to know. Keep others at a distance. 2. Conscientiousness: Waste my time. Leave my belongings around. 3. Agreeableness: Am not interested in other people’s problems. Insult people. 4. Openness: Am quick to understand things. Like to solve complex problems. 5. Neuroticism: Rarely get irritated. Keep my emotions under control. Seldom feel blue.",
    "----+": "1. Extraversion: Am hard to get to know. Keep others at a distance. 2. Conscientiousness: Waste my time. Leave my belongings around. 3. Agreeableness: Am not interested in other people’s problems. Insult people. 4. Openness: Have difficulty understanding abstract ideas. Seldom get lost in thought. 5. Neuroticism: Get angry easily. Get upset easily. Can be stirred up easily.",
    "-----": "1. Extraversion: Am hard to get to know. Keep others at a distance. 2. Conscientiousness: Waste my time. Leave my belongings around. 3. Agreeableness: Am not interested in other people’s problems. Insult people. 4. Openness: Have difficulty understanding abstract ideas. Seldom get lost in thought. 5. Neuroticism: Rarely get irritated. Keep my emotions under control. Seldom feel blue."
}


# ----------------------------------------------------------------------
# ★ 実験パラメータ
# ----------------------------------------------------------------------
# 検証したい性格特性のリスト
TARGET_TRAITS = ["++++-", "-++++", "-+++-", "----+", "+----", "+---+"]

# 1回の実行あたりのMMLU問題数
N_QUESTIONS = 50

# 各性格での繰り返し実行回数
N_REPEATS = 50


# ----------------------------------------------------------------------
# CLI (簡略化)
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run repeated single-agent MMLU experiments for statistical analysis.")
    p.add_argument("--seed", type=int, default=0, help="乱数シードのベース値 (-1 で毎回ランダム)")
    p.add_argument("--results-dir", default="results", help="保存先ルートディレクトリ")
    return p.parse_args()


# ----------------------------------------------------------------------
# 1 性格・1試行分の評価処理 (ほぼ変更なし)
# ----------------------------------------------------------------------
def eval_one_run(
    llama: Llama,
    ds,
    trait_code: str,
    run_id: str,        # 実験全体のID
    repetition_id: int, # 繰り返しID
    base_results: Path,
    seed: int,
) -> dict:
    """1回のMMLUタスク(50問)を実行し、正答率を返す"""
    persona_text = BIGFIVE.get(trait_code) # NONEの場合はNoneが返る
    trait_dir = trait_code

    # -------- logger とエージェント ----------
    # ログは試行ごとにサブディレクトリに保存 (例: results/RUN_ID/TRAIT/rep_0)
    logger = ExperimentLogger(
        team=trait_dir,
        seed=seed,
        model_path=llama.model_path,
        run_id=f"{run_id}/rep_{repetition_id}", # ログパスを試行ごとにユニークにする
        base_dir=Path(base_results),
    )
    agent = LlamaAgent(
        name=f"Agent_{trait_dir}",
        personality_text=persona_text,
        model=llama,
        max_tokens=256,
    )

    # -------- 問題ループ ----------
    correct_cnt = 0
    for idx in range(len(ds)):
        sample = ds[idx]
        task_info = sample["task_info"]
        gt_ans    = sample["answer"]

        if isinstance(task_info, (list, tuple)) and len(task_info) >= 2:
            q_txt, *opts = task_info
            choices = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(opts))
            question_block = f"Question {idx+1}:\n{q_txt}\n{choices}"
        else:
            question_block = f"Question {idx+1}:\n{task_info}"

        agent.reset_history()
        prompt = (
            question_block + "\n\n"
            "Respond ONLY in JSON as "
            '{"reasoning": "<Why you chose that answer>", "answer": "A/B/C/D"}'
        )
        resp = agent.generate_response(prompt, enforce_json=True)
        model_ans = resp.get("answer", "").strip()

        logger.log_turn({
            "q_id": idx + 1, "turn": 1, "agent": agent.name,
            "reasoning": resp.get("reasoning", ""), "answer": model_ans,
        })

        is_correct = int(model_ans == gt_ans)
        correct_cnt += is_correct

        logger.log_metric({
            "q_id": idx + 1, "final_answer": model_ans, "correct_answer": gt_ans,
            "correct": is_correct, "total_tokens": 0,
        })

    acc = correct_cnt / len(ds)
    # ターミナルに進捗を表示
    print(f"  Repetition {repetition_id+1}/{N_REPEATS} | Accuracy: {acc:.3%}")

    # この回の結果を返す
    return { "accuracy": acc }


# ----------------------------------------------------------------------
# main (★ ロジックを大幅に変更)
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    base_seed = args.seed
    if base_seed == -1:
        base_seed = random.randint(0, 10000)
    print(f"Using base seed: {base_seed}")

    run_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_root = Path(args.results_dir)

    # -------- モデルとデータは最初に一度だけ読み込む --------
    print("Loading LLaMA model...")
    llama = Llama(
        model_path=config.get_model_path(),
        chat_format="llama-3",
        n_ctx=65536,
        n_threads=8,
        n_gpu_layers=-1,
        verbose=False, # 大量実行のためモデルのログは抑制
    )
    print("Loading MMLU dataset...")
    ds = dataloader("mmlu", n_case=N_QUESTIONS)
    ds.set_mode("all")

    # -------- 全性格・全試行の正答率を格納する辞書 --------
    all_accuracies = {trait: [] for trait in TARGET_TRAITS}

    for trait_code in TARGET_TRAITS:
        print(f"\n===== Evaluating Trait: {trait_code} =====")
        
        for i in range(N_REPEATS):
            # 試行ごとにシードを設定し、再現性を確保
            current_seed = base_seed + i
            random.seed(current_seed)
            np.random.seed(current_seed)

            run_summary = eval_one_run(
                llama=llama,
                ds=ds,
                trait_code=trait_code,
                run_id=run_id,
                repetition_id=i,
                base_results=results_root,
                seed=current_seed,
            )
            all_accuracies[trait_code].append(run_summary["accuracy"])

    # -------- 最終的な統計結果の計算と出力 --------
    print("\n\n===== Experiment Finished: Summary Statistics =====")
    final_stats = []
    for trait_code, acc_list in all_accuracies.items():
        mean_acc = np.mean(acc_list)
        std_dev_acc = np.std(acc_list)
        
        final_stats.append({
            "trait": trait_code,
            "mean_accuracy": mean_acc,
            "std_dev_accuracy": std_dev_acc,
            "n_questions": N_QUESTIONS,
            "n_repeats": N_REPEATS,
        })
        print(f"Trait: {trait_code}")
        print(f"  - Mean Accuracy   : {mean_acc:.4f} ({mean_acc:.2%})")
        print(f"  - Std. Deviation  : {std_dev_acc:.4f}")

    # -------- 統計サマリーをCSVファイルに保存 --------
    # run_id 直下に保存
    summary_path = results_root / run_id
    summary_path.mkdir(parents=True, exist_ok=True)
    summary_csv_path = summary_path / "experiment_statistics.csv"

    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["trait", "mean_accuracy", "std_dev_accuracy", "n_questions", "n_repeats"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_stats)

    print("\n★ Final statistics saved to:", summary_csv_path.resolve())


if __name__ == "__main__":
    main()