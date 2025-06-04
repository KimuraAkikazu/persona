#!/usr/bin/env python
# calculate_team_state_change_rates.py
#
# エージェント 3 人の MMLU 回答ログ (T1〜T3) から
#   ・ポジティブ変化率 (T1 で多数派誤答 → T3 で多数派正答)
#   ・ネガティブ変化率 (T1 で多数派正答 → T3 で多数派誤答)
# を run／チーム単位で集計し、CSV に書き出すスクリプト。
#
# 2025-06-04: `groupby().agg()` を NamedAgg 依存から
# タプル構文へ置き換え、pandas 3.x でも動くよう修正。

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    """コマンドライン引数を解析する"""
    p = argparse.ArgumentParser(
        description=(
            "Calculates team-level Positive / Negative Change Rates and lists"
            " Q_IDs where all-incorrect@T1 → majority-correct@T3."
        )
    )
    p.add_argument(
        "--analysis-dir",
        default="analysis",
        help="分析結果を格納しているディレクトリ (project からの相対パス)",
    )
    p.add_argument("--run-id", required=True, help="対象 run_id (例: 20250521_011455)")
    p.add_argument(
        "--input-detailed-filename",
        default="answer_change_detailed_analysis_v2.csv",
        help="詳細分析 CSV ファイル名 (default: answer_change_detailed_analysis_v2.csv)",
    )
    p.add_argument(
        "--output-rates-filename",
        default="team_state_change_rates.csv",
        help="run ごとの集計結果を出力する CSV (default: team_state_change_rates.csv)",
    )
    p.add_argument(
        "--output-specific-qids-filename",
        default="t1_all_incorrect_to_t3_maj_correct_qids.csv",
        help="T1 全員誤答→T3 多数派正答となった Q_ID 一覧 CSV",
    )
    p.add_argument("--verbose", action="store_true", help="詳細ログを表示")
    return p.parse_args()


# ---------------------------------------------------------------------
# 便利関数
# ---------------------------------------------------------------------
def get_majority_answer_and_correctness(answers, correct_answer):
    """回答リストから多数派回答とそれが正解かどうかを返す"""
    if not answers or all(pd.isna(a) for a in answers):
        return None, 0, None

    valid = [a for a in answers if pd.notna(a)]
    if not valid:
        return None, 0, None

    ans, cnt = Counter(valid).most_common(1)[0]
    if cnt < 2:  # 3 人中 2 票未満 → 多数派なし
        return None, 0, None

    is_correct = ans == correct_answer if pd.notna(correct_answer) else None
    return ans, cnt, is_correct


def get_final_agent_answer(row):
    """T3→T2→T1 の優先順位で最終回答を取得"""
    for col in ("answer_t3", "answer_t2", "answer_t1"):
        if pd.notna(row[col]):
            return row[col]
    return None


# ---------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    base_path = Path(".") / args.analysis_dir / args.run_id

    src_csv = base_path / args.input_detailed_filename
    dst_rates = base_path / args.output_rates_filename
    dst_qids = base_path / args.output_specific_qids_filename

    if args.verbose:
        print(f"[INFO]  詳細 CSV: {src_csv.resolve()}")
        print(f"[INFO]  出力(rates): {dst_rates.resolve()}")
        print(f"[INFO]  出力(Q_ID):  {dst_qids.resolve()}")

    if not src_csv.exists():
        raise FileNotFoundError(f"入力 CSV が見つかりません: {src_csv}")

    df = pd.read_csv(src_csv)
    if df.empty:
        raise ValueError("入力 CSV が空です")

    required = {
        "run_id_timestamp",
        "team_name",
        "run_name",
        "q_id",
        "agent",
        "answer_t1",
        "answer_t2",
        "answer_t3",
        "correct_answer",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"CSV に必要な列が不足: {', '.join(sorted(missing))}")

    # q_id を整数にそろえる
    df["q_id"] = pd.to_numeric(df["q_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["q_id"]).reset_index(drop=True)
    df["q_id"] = df["q_id"].astype(int)

    run_results = []            # run ごとの変化率
    specific_events = []        # T1 全員誤答 → T3 多数派正答

    # -----------------------------------------------------------------
    # run × team ループ
    # -----------------------------------------------------------------
    for (ts, team, run), g in df.groupby(
        ["run_id_timestamp", "team_name", "run_name"]
    ):
        if args.verbose:
            print(f"\n[RUN] {ts} / {team} / {run}")

        pos_denom = pos_num = 0
        neg_denom = neg_num = 0
        cnt_all_incorrect_to_maj_corr = 0
        q_t1_maj_inc_no_final_maj = q_t1_maj_corr_no_final_maj = 0

        # -------------------------------
        # 各 Q_ID
        # -------------------------------
        for q_id, gq in g.groupby("q_id"):
            if len(gq["agent"].unique()) < 3:
                continue

            correct = gq["correct_answer"].iloc[0]
            if pd.isna(correct):
                continue

            # ---------------------------
            # T1 フェーズ
            # ---------------------------
            t1_answers = list(gq["answer_t1"].dropna())
            if len(t1_answers) < 3:
                continue

            num_correct_t1 = sum(a == correct for a in t1_answers)

            # ---------------------------
            # 最終回答 (T3→T2→T1)
            # ---------------------------
            final_answers = [get_final_agent_answer(row) for _, row in gq.iterrows()]
            if any(pd.isna(a) for a in final_answers):
                continue

            maj_ans, _, maj_is_correct = get_majority_answer_and_correctness(
                final_answers, correct
            )

            t1_maj_incorrect = num_correct_t1 <= 1
            t1_maj_correct = num_correct_t1 >= 2
            t1_all_incorrect = num_correct_t1 == 0

            # ---------------------------
            # ポジティブ変化率
            # ---------------------------
            if t1_maj_incorrect:
                pos_denom += 1
                if maj_ans is None:
                    q_t1_maj_inc_no_final_maj += 1
                elif maj_is_correct:
                    pos_num += 1
                    if t1_all_incorrect:
                        cnt_all_incorrect_to_maj_corr += 1
                        specific_events.append(
                            {
                                "run_id_timestamp": ts,
                                "team_name": team,
                                "run_name": run,
                                "q_id": q_id,
                                "t1_answers": ", ".join(sorted(t1_answers)),
                                "num_correct_t1": num_correct_t1,
                                "final_answers": ", ".join(sorted(final_answers)),
                                "team_final_majority_ans": maj_ans,
                                "correct_answer": correct,
                                "event": "T1_All_Incorrect_to_T3_Majority_Correct",
                            }
                        )

            # ---------------------------
            # ネガティブ変化率
            # ---------------------------
            if t1_maj_correct:
                neg_denom += 1
                if maj_ans is None:
                    q_t1_maj_corr_no_final_maj += 1
                elif maj_is_correct is False:
                    neg_num += 1

        run_results.append(
            {
                "run_id_timestamp": ts,
                "team_name": team,
                "run_name": run,
                # ポジティブ
                "denom_positive_change": pos_denom,
                "num_positive_change": pos_num,
                "positive_change_rate": pos_num / pos_denom if pos_denom else 0,
                # ネガティブ
                "denom_negative_change": neg_denom,
                "num_negative_change": neg_num,
                "negative_change_rate": neg_num / neg_denom if neg_denom else 0,
                # その他
                "count_all_incorrect_t1_to_maj_correct_t3_in_run": cnt_all_incorrect_to_maj_corr,
                "q_t1_maj_inc_no_final_maj": q_t1_maj_inc_no_final_maj,
                "q_t1_maj_corr_no_final_maj": q_t1_maj_corr_no_final_maj,
            }
        )

    # =================================================================
    # ① run ごとの CSV
    # =================================================================
    df_run = pd.DataFrame(run_results)
    df_run.to_csv(dst_rates, index=False, encoding="utf-8-sig")
    if args.verbose:
        print(f"[OK] run ごとの変化率を書き出しました → {dst_rates}")

    # =================================================================
    # ② team まとめ (平均・合計)
    #    タプル構文で集計することで pandas 3.x でも安全
    # =================================================================
    if not df_run.empty:
        df_team = (
            df_run.groupby(["run_id_timestamp", "team_name"], as_index=False)
            .agg(
                mean_positive_change_rate=("positive_change_rate", "mean"),
                std_positive_change_rate=("positive_change_rate", "std"),
                mean_negative_change_rate=("negative_change_rate", "mean"),
                std_negative_change_rate=("negative_change_rate", "std"),
                # 合計 (denom / num / count)
                sum_denom_positive_change=("denom_positive_change", "sum"),
                sum_num_positive_change=("num_positive_change", "sum"),
                sum_denom_negative_change=("denom_negative_change", "sum"),
                sum_num_negative_change=("num_negative_change", "sum"),
                sum_count_all_incorrect_t1_to_maj_correct_t3=(
                    "count_all_incorrect_t1_to_maj_correct_t3_in_run",
                    "sum",
                ),
                sum_q_t1_maj_inc_no_final_maj=("q_t1_maj_inc_no_final_maj", "sum"),
                sum_q_t1_maj_corr_no_final_maj=("q_t1_maj_corr_no_final_maj", "sum"),
                num_runs_analyzed=("run_name", "nunique"),
            )
        )

        # 合計値からの再計算率
        df_team["overall_positive_change_rate_from_totals"] = df_team.apply(
            lambda r: r["sum_num_positive_change"] / r["sum_denom_positive_change"]
            if r["sum_denom_positive_change"]
            else 0,
            axis=1,
        )
        df_team["overall_negative_change_rate_from_totals"] = df_team.apply(
            lambda r: r["sum_num_negative_change"] / r["sum_denom_negative_change"]
            if r["sum_denom_negative_change"]
            else 0,
            axis=1,
        )

        dst_team = base_path / "team_state_change_rates_avg_summary.csv"
        df_team.to_csv(dst_team, index=False, encoding="utf-8-sig")
        if args.verbose:
            print(f"[OK] チーム平均まとめを書き出しました → {dst_team}")

    # =================================================================
    # ③ 特定 Q_ID 一覧
    # =================================================================
    if specific_events:
        pd.DataFrame(specific_events).to_csv(
            dst_qids, index=False, encoding="utf-8-sig"
        )
        if args.verbose:
            print(
                f"[OK] T1 全員誤答→T3 多数派正答の Q_ID を書き出しました ({len(specific_events)} 件)"
            )


if __name__ == "__main__":
    main()
