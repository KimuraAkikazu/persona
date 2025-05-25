#!/usr/bin/env python
# analyze_answer_changes.py

import argparse
import pandas as pd
from pathlib import Path
import json
import pickle # For loading MMLU answers
import numpy as np

DEFAULT_MMLU_FILE_PATH = "eval_data/mmlu.pkl"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyzes agent answer changes per turn, including correctness and overall statistics."
    )
    parser.add_argument(
        "--analysis-dir",
        default="analysis",
        help="分析データが格納されているディレクトリ (default: analysis, projectディレクトリからの相対パス)",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="分析対象のrun_id (タイムスタンプ付きのフォルダ名)",
    )
    parser.add_argument(
        "--input-answers-log-filename", # 前のスクリプトからの入力ファイル名
        default="all_answers_log.csv",
        help="エージェントの全回答ログが記録されたCSVファイル名 (default: all_answers_log.csv)",
    )
    parser.add_argument(
        "--output-detailed-filename",
        default="answer_change_detailed_analysis_v2.csv",
        help="正誤分析を含む詳細な回答変更ログを出力するCSVファイル名 (default: answer_change_detailed_analysis_v2.csv)",
    )
    parser.add_argument(
        "--output-per-turn-summary-filename",
        default="answer_change_per_turn_summary.csv",
        help="ターンごとの変更パターンの集計結果を出力するCSVファイル名 (default: answer_change_per_turn_summary.csv)",
    )
    parser.add_argument( # 新しい出力ファイル名
        "--output-agent-metrics-filename",
        default="agent_change_metrics_summary.csv",
        help="エージェントごとの回答変更メトリクスの平均等を出力するCSVファイル名 (default: agent_change_metrics_summary.csv)",
    )
    parser.add_argument(
        "--mmlu-answers-path",
        default=DEFAULT_MMLU_FILE_PATH,
        help=f"MMLUの正解データファイルへのパス (default: {DEFAULT_MMLU_FILE_PATH}, projectディレクトリからの相対パス)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細なログ出力を有効にする"
    )
    return parser.parse_args()

def load_mmlu_answers(mmlu_path: Path, verbose: bool = False):
    # (この関数は変更なし)
    if not mmlu_path.exists():
        if verbose:
            print(f"警告: MMLU正解データファイルが見つかりません: {mmlu_path.resolve()}")
        return None
    try:
        with open(mmlu_path, "rb") as f:
            data = pickle.load(f)
            correct_answers = {
                (i + 1): ans for i, ans in enumerate(data.get("answer", []))
            }
            if not correct_answers:
                if verbose:
                    print(f"警告: MMLU正解データファイル {mmlu_path.resolve()} から正解情報を読み込めませんでした。'answer'キーがないか空です。")
                return None
            if verbose:
                print(f"MMLU正解データを正常にロードしました。件数: {len(correct_answers)}")
            return correct_answers
    except Exception as e:
        if verbose:
            print(f"エラー: MMLU正解データファイルのロード中にエラーが発生しました: {mmlu_path.resolve()}, 詳細: {e}")
        return None

def get_change_pattern(ans_from, ans_to, correct_answer):
    # (この関数は変更なし)
    if pd.isna(ans_from) or pd.isna(ans_to):
        if pd.isna(ans_from) and pd.isna(ans_to): return "NoAnswer_Both"
        if pd.isna(ans_from): return "NoAnswer_From"
        if pd.isna(ans_to): return "NoAnswer_To"

    if pd.isna(correct_answer):
        if ans_from == ans_to:
            return "NoChange_CorrectAnswerMissing"
        else:
            return "Changed_CorrectAnswerMissing"

    from_is_correct = (ans_from == correct_answer)
    to_is_correct = (ans_to == correct_answer)

    if ans_from == ans_to:
        return "NoChange_Correct" if from_is_correct else "NoChange_Incorrect"
    else:
        if from_is_correct and not to_is_correct:
            return "CtoI"
        elif not from_is_correct and to_is_correct:
            return "ItoC"
        elif not from_is_correct and not to_is_correct:
            return "ItoI_diff"
        elif from_is_correct and to_is_correct:
            return "CtoC_diff"
        else:
             return "Error_Pattern"

def main():
    args = parse_args()
    current_dir = Path(".")
    base_analysis_path = current_dir / args.analysis_dir
    run_analysis_path = base_analysis_path / args.run_id

    input_csv_file = run_analysis_path / args.input_answers_log_filename # 修正: 引数名変更
    output_detailed_file = run_analysis_path / args.output_detailed_filename
    output_per_turn_summary_file = run_analysis_path / args.output_per_turn_summary_filename # 修正: 引数名変更
    output_agent_metrics_file = run_analysis_path / args.output_agent_metrics_filename # 新しい出力ファイル
    mmlu_answers_path = current_dir / args.mmlu_answers_path

    if args.verbose:
        print(f"--- Agent Answer Change Metrics Analysis Started for Run ID: {args.run_id} ---")
        print(f"Input Answers Log CSV from: {input_csv_file.resolve()}")
        print(f"MMLU Answers from: {mmlu_answers_path.resolve()}")
        print(f"Output Detailed CSV to: {output_detailed_file.resolve()}")
        print(f"Output Per-Turn Summary CSV to: {output_per_turn_summary_file.resolve()}")
        print(f"Output Agent Metrics CSV to: {output_agent_metrics_file.resolve()}")


    # --- ステップ1: 詳細な変更ログCSV (answer_change_detailed_analysis_v2.csv) の生成 ---
    # この部分は前のコードとほぼ同じ
    if not input_csv_file.exists():
        print(f"エラー: 入力となる全回答ログCSVファイルが見つかりません: {input_csv_file.resolve()}")
        return

    mmlu_correct_answers = load_mmlu_answers(mmlu_answers_path, args.verbose)

    try:
        df_answers_log = pd.read_csv(input_csv_file)
    except Exception as e:
        print(f"エラー: 入力CSVファイルの読み込み中にエラーが発生しました: {input_csv_file.resolve()}, 詳細: {e}")
        return

    if df_answers_log.empty:
        print(f"入力CSVファイル {input_csv_file.resolve()} が空です。")
        return

    required_columns = ["run_id_timestamp", "team_name", "run_name", "q_id", "agent", "turn", "answer"]
    if not all(col in df_answers_log.columns for col in required_columns):
        print(f"エラー: 入力CSVファイルに必要な列が含まれていません。必要な列: {', '.join(required_columns)}")
        return

    df_answers_log['turn'] = pd.to_numeric(df_answers_log['turn'], errors='coerce').dropna().astype(int)
    df_answers_log['q_id'] = pd.to_numeric(df_answers_log['q_id'], errors='coerce').dropna().astype(int)

    pivot_df_detailed = df_answers_log.pivot_table(
        index=['run_id_timestamp', 'team_name', 'run_name', 'q_id', 'agent'],
        columns='turn', values='answer', aggfunc='first'
    ).reset_index()
    pivot_df_detailed.rename(columns={1: 'answer_t1', 2: 'answer_t2', 3: 'answer_t3'}, inplace=True)

    if mmlu_correct_answers:
        correct_answers_df = pd.Series(mmlu_correct_answers, name='correct_answer').rename_axis('q_id').reset_index()
        pivot_df_detailed = pd.merge(pivot_df_detailed, correct_answers_df, on='q_id', how='left')
    else:
        pivot_df_detailed['correct_answer'] = None

    pivot_df_detailed['change_pattern_t1_t2'] = pivot_df_detailed.apply(
        lambda row: get_change_pattern(row['answer_t1'], row.get('answer_t2'), row['correct_answer']), axis=1
    )
    pivot_df_detailed['change_pattern_t2_t3'] = pivot_df_detailed.apply(
        lambda row: get_change_pattern(row.get('answer_t2'), row.get('answer_t3'), row['correct_answer']), axis=1
    )
    pivot_df_detailed['changed_t1_t2'] = ((pivot_df_detailed['answer_t1'] != pivot_df_detailed['answer_t2']) & pd.notna(pivot_df_detailed['answer_t1']) & pd.notna(pivot_df_detailed['answer_t2'])).astype(int)
    pivot_df_detailed['changed_t2_t3'] = ((pivot_df_detailed['answer_t2'] != pivot_df_detailed['answer_t3']) & pd.notna(pivot_df_detailed['answer_t2']) & pd.notna(pivot_df_detailed['answer_t3'])).astype(int)

    try:
        pivot_df_detailed.to_csv(output_detailed_file, index=False, encoding="utf-8-sig")
        if args.verbose:
            print(f"\n詳細な回答変更ログがCSVファイルに保存されました: {output_detailed_file.resolve()}")
    except Exception as e:
        print(f"エラー: 詳細ログCSVファイルへの書き込み中にエラーが発生しました: {output_detailed_file.resolve()}, 詳細: {e}")


    # --- ステップ2: ターンごとの変更パターン集計CSV (answer_change_per_turn_summary.csv) の生成 ---
    # この部分も前のコードとほぼ同じ
    summary_melt_t1_t2 = pivot_df_detailed.groupby(
        ['run_id_timestamp', 'team_name', 'run_name', 'agent', 'change_pattern_t1_t2']
    ).size().reset_index(name='count')
    summary_melt_t1_t2.rename(columns={'change_pattern_t1_t2': 'change_pattern', 'count': 'count_t1_t2'}, inplace=True)

    summary_melt_t2_t3 = pivot_df_detailed.groupby(
        ['run_id_timestamp', 'team_name', 'run_name', 'agent', 'change_pattern_t2_t3']
    ).size().reset_index(name='count')
    summary_melt_t2_t3.rename(columns={'change_pattern_t2_t3': 'change_pattern', 'count': 'count_t2_t3'}, inplace=True)
    
    all_agents_runs_df = pivot_df_detailed[['run_id_timestamp', 'team_name', 'run_name', 'agent']].drop_duplicates()
    
    summary_t1_t2_pivot = summary_melt_t1_t2.pivot_table(
        index=['run_id_timestamp', 'team_name', 'run_name', 'agent'], columns='change_pattern', values='count_t1_t2', fill_value=0
    ).add_suffix('_t1_t2').reset_index()

    summary_t2_t3_pivot = summary_melt_t2_t3.pivot_table(
        index=['run_id_timestamp', 'team_name', 'run_name', 'agent'], columns='change_pattern', values='count_t2_t3', fill_value=0
    ).add_suffix('_t2_t3').reset_index()

    df_per_turn_summary = pd.merge(all_agents_runs_df, summary_t1_t2_pivot, on=['run_id_timestamp', 'team_name', 'run_name', 'agent'], how='left')
    df_per_turn_summary = pd.merge(df_per_turn_summary, summary_t2_t3_pivot, on=['run_id_timestamp', 'team_name', 'run_name', 'agent'], how='left')
    df_per_turn_summary.fillna(0, inplace=True)

    try:
        df_per_turn_summary.to_csv(output_per_turn_summary_file, index=False, encoding="utf-8-sig")
        if args.verbose:
            print(f"\nターンごとの回答変更パターンの集計結果がCSVファイルに保存されました: {output_per_turn_summary_file.resolve()}")
    except Exception as e:
        print(f"エラー: ターンごとの集計結果CSVファイルへの書き込み中にエラーが発生しました: {output_per_turn_summary_file.resolve()}, 詳細: {e}")


    # --- ステップ3: エージェントごとの回答変更メトリクス集計 (新しい処理) ---
    if df_per_turn_summary.empty:
        print("ターンごとの集計データが空のため、エージェントメトリクスの集計はスキップします。")
    else:
        # 各run, 各agentでの総変更回数、ItoC回数、CtoI回数を計算
        # _t1_t2 と _t2_t3 の両方で発生した変更をカウントする
        
        # 総変更回数: "NoChange_" で始まらないパターンの合計
        change_pattern_columns_t1_t2 = [col for col in df_per_turn_summary.columns if col.endswith('_t1_t2') and not col.startswith('NoChange_') and not col.startswith('NoAnswer_')]
        change_pattern_columns_t2_t3 = [col for col in df_per_turn_summary.columns if col.endswith('_t2_t3') and not col.startswith('NoChange_') and not col.startswith('NoAnswer_')]

        df_per_turn_summary['total_changes_per_run'] = df_per_turn_summary[change_pattern_columns_t1_t2].sum(axis=1) + \
                                                       df_per_turn_summary[change_pattern_columns_t2_t3].sum(axis=1)
        
        # ItoC回数 (列名が存在する場合のみ加算)
        itoc_cols_t1_t2 = [col for col in ['ItoC_t1_t2'] if col in df_per_turn_summary.columns]
        itoc_cols_t2_t3 = [col for col in ['ItoC_t2_t3'] if col in df_per_turn_summary.columns]
        df_per_turn_summary['itoc_changes_per_run'] = (df_per_turn_summary[itoc_cols_t1_t2].sum(axis=1) if itoc_cols_t1_t2 else 0) + \
                                                      (df_per_turn_summary[itoc_cols_t2_t3].sum(axis=1) if itoc_cols_t2_t3 else 0)
        
        # CtoI回数 (列名が存在する場合のみ加算)
        ctoi_cols_t1_t2 = [col for col in ['CtoI_t1_t2'] if col in df_per_turn_summary.columns]
        ctoi_cols_t2_t3 = [col for col in ['CtoI_t2_t3'] if col in df_per_turn_summary.columns]
        df_per_turn_summary['ctoi_changes_per_run'] = (df_per_turn_summary[ctoi_cols_t1_t2].sum(axis=1) if ctoi_cols_t1_t2 else 0) + \
                                                      (df_per_turn_summary[ctoi_cols_t2_t3].sum(axis=1) if ctoi_cols_t2_t3 else 0)

        if args.verbose:
            print("\nSample of per_turn_summary with per-run change counts:")
            print(df_per_turn_summary[['run_name', 'agent', 'total_changes_per_run', 'itoc_changes_per_run', 'ctoi_changes_per_run']].head())

        # チームごと、エージェントごとに平均を算出
        agent_metrics_summary = df_per_turn_summary.groupby(['run_id_timestamp', 'team_name', 'agent']).agg(
            avg_total_changes_per_run=('total_changes_per_run', 'mean'),
            std_total_changes_per_run=('total_changes_per_run', 'std'),
            avg_itoc_changes_per_run=('itoc_changes_per_run', 'mean'),
            std_itoc_changes_per_run=('itoc_changes_per_run', 'std'),
            avg_ctoi_changes_per_run=('ctoi_changes_per_run', 'mean'),
            std_ctoi_changes_per_run=('ctoi_changes_per_run', 'std'),
            num_runs_for_agent=('run_name', 'nunique')
        ).reset_index()

        # NaNを0で埋める (stdが1つのrunしかない場合にNaNになるため)
        agent_metrics_summary.fillna({
            'std_total_changes_per_run': 0,
            'std_itoc_changes_per_run': 0,
            'std_ctoi_changes_per_run': 0
        }, inplace=True)

        if args.verbose:
            print("\nSample of agent_metrics_summary:")
            print(agent_metrics_summary.head())

        if not agent_metrics_summary.empty:
            try:
                agent_metrics_summary.to_csv(output_agent_metrics_file, index=False, encoding="utf-8-sig")
                print(f"\nエージェントごとの回答変更メトリクスの集計結果がCSVファイルに保存されました: {output_agent_metrics_file.resolve()}")
            except Exception as e:
                print(f"エラー: エージェントメトリクス集計CSVファイルへの書き込み中にエラーが発生しました: {output_agent_metrics_file.resolve()}, 詳細: {e}")
        else:
            print("エージェントごとの回答変更メトリクスの集計データがありませんでした。")


    if args.verbose:
        print(f"--- Agent Answer Change Metrics Analysis for Run ID: {args.run_id} Complete ---")

if __name__ == "__main__":
    main()