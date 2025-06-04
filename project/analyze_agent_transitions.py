#!/usr/bin/env python
# analyze_agent_transitions.py

import argparse
import pandas as pd
from pathlib import Path
import numpy as np

def parse_args():
    """コマンドライン引数を解析します."""
    parser = argparse.ArgumentParser(
        description="Analyzes individual agent answer transition patterns from detailed logs."
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
        "--input-detailed-filename",
        default="answer_change_detailed_analysis_v2.csv",
        help="入力する詳細分析CSVファイル名 (default: answer_change_detailed_analysis_v2.csv)",
    )
    parser.add_argument(
        "--output-agent-profile-filename",
        default="agent_answer_transition_profile.csv",
        help="エージェントごとの回答変遷プロファイルを出力するCSVファイル名 (default: agent_answer_transition_profile.csv)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細なログ出力を有効にする"
    )
    return parser.parse_args()

def get_final_answer(row):
    """T3, T2, T1 の優先順位で最終回答を取得する"""
    if pd.notna(row['answer_t3']):
        return row['answer_t3']
    elif pd.notna(row['answer_t2']):
        return row['answer_t2']
    elif pd.notna(row['answer_t1']):
        return row['answer_t1']
    return None

def main():
    args = parse_args()
    current_dir = Path(".") # /project ディレクトリを想定

    base_analysis_path = current_dir / args.analysis_dir
    run_analysis_path = base_analysis_path / args.run_id

    input_detailed_csv_file = run_analysis_path / args.input_detailed_filename
    output_agent_profile_csv_file = run_analysis_path / args.output_agent_profile_filename

    if args.verbose:
        print(f"--- Agent Answer Transition Profile Analysis Started for Run ID: {args.run_id} ---")
        print(f"Input Detailed CSV from: {input_detailed_csv_file.resolve()}")
        print(f"Output Agent Profile CSV to: {output_agent_profile_csv_file.resolve()}")

    if not input_detailed_csv_file.exists():
        print(f"エラー: 入力となる詳細分析CSVファイルが見つかりません: {input_detailed_csv_file.resolve()}")
        return

    try:
        df_detailed = pd.read_csv(input_detailed_csv_file)
    except Exception as e:
        print(f"エラー: 入力CSVファイルの読み込み中にエラーが発生しました: {input_detailed_csv_file.resolve()}, 詳細: {e}")
        return

    if df_detailed.empty:
        print(f"入力CSVファイル {input_detailed_csv_file.resolve()} が空です。")
        return

    # 必要な列の確認
    required_cols = ['run_id_timestamp', 'team_name', 'run_name', 'q_id', 'agent',
                     'answer_t1', 'answer_t2', 'answer_t3', 'correct_answer',
                     'change_pattern_t1_t2', 'change_pattern_t2_t3']
    if not all(col in df_detailed.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df_detailed.columns]
        print(f"エラー: 入力CSVファイルに必要な列が不足しています。不足している列: {', '.join(missing_cols)}")
        return

    # --- エージェントごとの分析 ---
    # 最終回答を決定
    df_detailed['answer_final'] = df_detailed.apply(get_final_answer, axis=1)

    # 指標Bの計算のための準備
    agent_q_profiles = []

    for _, row in df_detailed.iterrows():
        profile = {
            'run_id_timestamp': row['run_id_timestamp'],
            'team_name': row['team_name'],
            'run_name': row['run_name'],
            'q_id': row['q_id'],
            'agent': row['agent'],
            'correct_answer': row['correct_answer'],
            'answer_t1': row['answer_t1'],
            'answer_final': row['answer_final'],
            'change_pattern_t1_t2': row['change_pattern_t1_t2'],
            'change_pattern_t2_t3': row['change_pattern_t2_t3']
        }

        ans1 = row['answer_t1']
        ans_final = row['answer_final']
        correct_ans = row['correct_answer']
        
        # 指標Bの分類
        profile['B_pattern'] = "Unknown"
        if pd.notna(ans1) and pd.notna(ans_final) and pd.notna(correct_ans):
            is_t1_correct = (ans1 == correct_ans)
            is_final_correct = (ans_final == correct_ans)
            is_changed_t1_final = (ans1 != ans_final)

            if not is_changed_t1_final: # 変更なし
                if is_t1_correct: # 必然的に is_final_correct も True
                    profile['B_pattern'] = "B1_Correct_Stay"
                else: # 必然的に is_final_correct も False
                    profile['B_pattern'] = "B2_Incorrect_Stay"
            else: # 変更あり
                if not is_t1_correct and is_final_correct:
                    profile['B_pattern'] = "B3_Self_Correction_ItoC"
                elif is_t1_correct and not is_final_correct:
                    profile['B_pattern'] = "B4_Error_Induction_CtoI"
                elif not is_t1_correct and not is_final_correct: # is_changed_t1_final が True なので ans1 != ans_final
                    profile['B_pattern'] = "B5_Incorrect_To_Different_Incorrect"
                elif is_t1_correct and is_final_correct: # is_changed_t1_final が True なので ans1 != ans_final
                    profile['B_pattern'] = "B6_Correct_To_Different_Correct" # 非常に稀
        elif pd.isna(correct_ans):
            profile['B_pattern'] = "CorrectAnswer_Missing"
        elif pd.isna(ans1) or pd.isna(ans_final):
             profile['B_pattern'] = "Answer_T1_or_Final_Missing"


        agent_q_profiles.append(profile)

    df_agent_q_profiles = pd.DataFrame(agent_q_profiles)

    if args.verbose:
        print("\nSample of agent question profiles with B_pattern:")
        print(df_agent_q_profiles[['run_name', 'q_id', 'agent', 'answer_t1', 'answer_final', 'correct_answer', 'B_pattern']].head())

    # --- 集計 ---
    # 1. 指標B (T1 vs Final) のパターンごとの回数を集計
    summary_b_patterns = df_agent_q_profiles.groupby(
        ['run_id_timestamp', 'team_name', 'agent', 'B_pattern']
    ).size().unstack(fill_value=0).reset_index()

    # 2. 指標C (T1->T2, T2->T3) のパターンごとの回数を集計
    summary_c_patterns_t1_t2 = df_agent_q_profiles.groupby(
        ['run_id_timestamp', 'team_name', 'agent', 'change_pattern_t1_t2']
    ).size().unstack(fill_value=0).add_suffix('_T1_T2').reset_index()

    summary_c_patterns_t2_t3 = df_agent_q_profiles.groupby(
        ['run_id_timestamp', 'team_name', 'agent', 'change_pattern_t2_t3']
    ).size().unstack(fill_value=0).add_suffix('_T2_T3').reset_index()

    # 3. 指標A (変更回数) の計算と、その他の基本情報
    # 質問総数
    total_questions_per_agent = df_agent_q_profiles.groupby(
        ['run_id_timestamp', 'team_name', 'agent']
    )['q_id'].nunique().reset_index(name='total_questions_answered_by_agent')

    # T1->Final 変更回数
    df_agent_q_profiles['changed_t1_final'] = (df_agent_q_profiles['answer_t1'] != df_agent_q_profiles['answer_final']) & \
                                              pd.notna(df_agent_q_profiles['answer_t1']) & \
                                              pd.notna(df_agent_q_profiles['answer_final'])
    
    changes_t1_final_count = df_agent_q_profiles.groupby(
        ['run_id_timestamp', 'team_name', 'agent']
    )['changed_t1_final'].sum().reset_index(name='A2_changes_t1_to_final')

    # T1->T2 または T2->T3 での変更回数 (トランジションベース)
    # change_pattern が "NoChange_" や "NoAnswer_" で始まらないものの数をカウント
    def count_transitions_changed(df_group, pattern_col_name):
        return df_group[~df_group[pattern_col_name].str.startswith('NoChange_', na=False) & \
                        ~df_group[pattern_col_name].str.startswith('NoAnswer_', na=False) & \
                        ~df_group[pattern_col_name].str.startswith('Changed_CorrectAnswerMissing', na=False) & \
                        ~df_group[pattern_col_name].str.startswith('CorrectAnswer_Missing', na=False) & \
                        ~df_group[pattern_col_name].str.startswith('Answer_T1_or_Final_Missing', na=False) & \
                        ~df_group[pattern_col_name].str.startswith('Error_Pattern', na=False)
                        ].shape[0]


    transitions_t1_t2_changed_count = df_agent_q_profiles.groupby(
        ['run_id_timestamp', 'team_name', 'agent']
    ).apply(lambda x: count_transitions_changed(x, 'change_pattern_t1_t2')).reset_index(name='A1_transitions_changed_t1_t2')
    
    transitions_t2_t3_changed_count = df_agent_q_profiles.groupby(
        ['run_id_timestamp', 'team_name', 'agent']
    ).apply(lambda x: count_transitions_changed(x, 'change_pattern_t2_t3')).reset_index(name='A1_transitions_changed_t2_t3')


    # 全てのサマリーをマージ
    agent_profile_summary_df = total_questions_per_agent
    agent_profile_summary_df = pd.merge(agent_profile_summary_df, changes_t1_final_count, on=['run_id_timestamp', 'team_name', 'agent'], how='left')
    agent_profile_summary_df = pd.merge(agent_profile_summary_df, transitions_t1_t2_changed_count, on=['run_id_timestamp', 'team_name', 'agent'], how='left')
    agent_profile_summary_df = pd.merge(agent_profile_summary_df, transitions_t2_t3_changed_count, on=['run_id_timestamp', 'team_name', 'agent'], how='left')
    agent_profile_summary_df = pd.merge(agent_profile_summary_df, summary_b_patterns, on=['run_id_timestamp', 'team_name', 'agent'], how='left')
    agent_profile_summary_df = pd.merge(agent_profile_summary_df, summary_c_patterns_t1_t2, on=['run_id_timestamp', 'team_name', 'agent'], how='left')
    agent_profile_summary_df = pd.merge(agent_profile_summary_df, summary_c_patterns_t2_t3, on=['run_id_timestamp', 'team_name', 'agent'], how='left')

    # NaNを0で埋める（集計結果がない場合など）
    agent_profile_summary_df.fillna(0, inplace=True)

    # 指標A1: 総変更回数 (トランジションの合計)
    agent_profile_summary_df['A1_total_transitions_changed'] = agent_profile_summary_df['A1_transitions_changed_t1_t2'] + agent_profile_summary_df['A1_transitions_changed_t2_t3']

    # 各指標Bのパターンを率に変換 (total_questions_answered_by_agent を分母とする)
    b_pattern_cols = [col for col in agent_profile_summary_df.columns if col.startswith('B') and col not in ['B_pattern']] # B_pattern 自体は除外
    for col in b_pattern_cols:
        agent_profile_summary_df[f'rate_{col}'] = agent_profile_summary_df.apply(
            lambda row: (row[col] / row['total_questions_answered_by_agent']) if row['total_questions_answered_by_agent'] > 0 else 0,
            axis=1
        )
    
    # 特定の重要な率を明示的にカラム名として保持
    if 'B3_Self_Correction_ItoC' in agent_profile_summary_df.columns:
         agent_profile_summary_df['self_correction_rate'] = agent_profile_summary_df['rate_B3_Self_Correction_ItoC']
    if 'B4_Error_Induction_CtoI' in agent_profile_summary_df.columns:
         agent_profile_summary_df['error_induction_rate_self'] = agent_profile_summary_df['rate_B4_Error_Induction_CtoI']


    if args.verbose:
        print("\nSample of final agent profile summary:")
        print(agent_profile_summary_df[['team_name', 'agent', 'total_questions_answered_by_agent', 'A1_total_transitions_changed', 'A2_changes_t1_to_final', 'self_correction_rate', 'error_induction_rate_self']].head())

    # 結果をCSVファイルに保存
    try:
        agent_profile_summary_df.to_csv(output_agent_profile_csv_file, index=False, encoding="utf-8-sig")
        print(f"\nエージェントごとの回答変遷プロファイルがCSVファイルに保存されました: {output_agent_profile_csv_file.resolve()}")
    except Exception as e:
        print(f"エラー: エージェントプロファイルCSVファイルへの書き込み中にエラーが発生しました: {output_agent_profile_csv_file.resolve()}, 詳細: {e}")

    if args.verbose:
        print(f"--- Agent Answer Transition Profile Analysis for Run ID: {args.run_id} Complete ---")

if __name__ == "__main__":
    main()