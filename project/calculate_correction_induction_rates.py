#!/usr/bin/env python
# calculate_correction_induction_rates.py

import argparse
import pandas as pd
from pathlib import Path
from collections import Counter

def parse_args():
    """コマンドライン引数を解析します."""
    parser = argparse.ArgumentParser(
        description="Calculates Answer Correction Rate and Error Induction Rate, categorized by initial correct answer counts."
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
        "--output-rates-filename",
        default="correction_induction_rates_summary_v2.csv",
        help="回答修正率と誤答誘発率の集計結果(初期正解者数別)を出力するCSVファイル名 (default: correction_induction_rates_summary_v2.csv)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細なログ出力を有効にする"
    )
    return parser.parse_args()

def get_majority_answer(answers):
    """回答リストから多数派の回答を返す。多数派がなければNoneを返す。"""
    if not answers or all(pd.isna(ans) for ans in answers):
        return None, 0
    
    valid_answers = [ans for ans in answers if pd.notna(ans)]
    if not valid_answers:
        return None, 0

    counts = Counter(valid_answers)
    if not counts:
        return None, 0
        
    most_common_ans = counts.most_common(1)
    if not most_common_ans:
        return None, 0

    majority_ans, majority_count = most_common_ans[0]

    if majority_count >= 2: # 3エージェント中2票以上
        return majority_ans, majority_count
    else:
        return None, 0


def main():
    args = parse_args()
    current_dir = Path(".")

    base_analysis_path = current_dir / args.analysis_dir
    run_analysis_path = base_analysis_path / args.run_id

    input_detailed_csv_file = run_analysis_path / args.input_detailed_filename
    output_rates_csv_file = run_analysis_path / args.output_rates_filename

    if args.verbose:
        print(f"--- Correction/Induction Rate Calculation (v2) Started for Run ID: {args.run_id} ---")
        print(f"Input Detailed CSV from: {input_detailed_csv_file.resolve()}")
        print(f"Output Rates CSV to: {output_rates_csv_file.resolve()}")

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

    required_cols = ['run_id_timestamp', 'team_name', 'run_name', 'q_id', 'agent',
                     'answer_t1', 'answer_t2', 'answer_t3', 'correct_answer']
    if not all(col in df_detailed.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df_detailed.columns]
        print(f"エラー: 入力CSVファイルに必要な列が不足しています。不足している列: {', '.join(missing_cols)}")
        return

    results_list = []

    for (run_id_ts, team_name, run_name), group_run_team in df_detailed.groupby(['run_id_timestamp', 'team_name', 'run_name']):
        if args.verbose:
            print(f"\nProcessing: Run ID={run_id_ts}, Team={team_name}, Run={run_name}")

        denom_disagree_t1correct1 = 0
        corr_num_disagree_t1correct1 = 0
        ind_num_disagree_t1correct1 = 0
        no_maj_final_disagree_t1correct1 = 0
        denom_disagree_t1correct2 = 0
        corr_num_disagree_t1correct2 = 0
        ind_num_disagree_t1correct2 = 0
        no_maj_final_disagree_t1correct2 = 0
        denom_disagree_overall = 0
        corr_num_disagree_overall = 0
        ind_num_disagree_overall = 0
        no_maj_final_disagree_overall = 0

        for q_id, group_q in group_run_team.groupby('q_id'):
            if len(group_q['agent'].unique()) < 3:
                if args.verbose:
                    print(f"  Skipping Q_ID {q_id}: Less than 3 agents' data found.")
                continue

            correct_ans_for_q = group_q['correct_answer'].iloc[0]
            if pd.isna(correct_ans_for_q):
                if args.verbose:
                    print(f"  Skipping Q_ID {q_id}: Correct answer is missing.")
                continue

            t1_answers_map = {}
            num_t1_correct_answers = 0
            valid_t1_answers_list = []

            for _, row in group_q.iterrows():
                agent = row['agent']
                ans_t1 = row['answer_t1']
                t1_answers_map[agent] = ans_t1
                if pd.notna(ans_t1):
                    valid_t1_answers_list.append(ans_t1)
                    if ans_t1 == correct_ans_for_q:
                        num_t1_correct_answers += 1
            
            if len(valid_t1_answers_list) < 3:
                 if args.verbose:
                    print(f"  Skipping Q_ID {q_id}: Not all 3 agents had valid T1 answers (found {len(valid_t1_answers_list)}).")
                 continue

            initial_disagreement = len(set(valid_t1_answers_list)) > 1

            if initial_disagreement:
                final_answers_for_q = []
                for _, row in group_q.iterrows():
                    ans_t3 = row['answer_t3']
                    ans_t2 = row['answer_t2']
                    ans_t1 = row['answer_t1']
                    
                    final_ans_agent = ans_t1
                    if pd.notna(ans_t3):
                        final_ans_agent = ans_t3
                    elif pd.notna(ans_t2):
                        final_ans_agent = ans_t2
                    final_answers_for_q.append(final_ans_agent)

                if len([fa for fa in final_answers_for_q if pd.notna(fa)]) < 3:
                    if num_t1_correct_answers == 1: no_maj_final_disagree_t1correct1 +=1
                    if num_t1_correct_answers == 2: no_maj_final_disagree_t1correct2 +=1
                    no_maj_final_disagree_overall +=1
                    continue

                team_majority_final_ans, _ = get_majority_answer(final_answers_for_q)

                denom_disagree_overall += 1
                if team_majority_final_ans is not None:
                    if team_majority_final_ans == correct_ans_for_q:
                        corr_num_disagree_overall += 1
                    else:
                        ind_num_disagree_overall += 1
                else:
                    no_maj_final_disagree_overall += 1

                if num_t1_correct_answers == 1:
                    denom_disagree_t1correct1 += 1
                    if team_majority_final_ans is not None:
                        if team_majority_final_ans == correct_ans_for_q:
                            corr_num_disagree_t1correct1 += 1
                        else:
                            ind_num_disagree_t1correct1 += 1
                    else:
                        no_maj_final_disagree_t1correct1 += 1
                
                elif num_t1_correct_answers == 2:
                    denom_disagree_t1correct2 += 1
                    if team_majority_final_ans is not None:
                        if team_majority_final_ans == correct_ans_for_q:
                            corr_num_disagree_t1correct2 += 1
                        else:
                            ind_num_disagree_t1correct2 += 1
                    else:
                        no_maj_final_disagree_t1correct2 += 1
        
        rate_corr_overall = (corr_num_disagree_overall / denom_disagree_overall) if denom_disagree_overall > 0 else 0
        rate_ind_overall = (ind_num_disagree_overall / denom_disagree_overall) if denom_disagree_overall > 0 else 0
        rate_corr_t1c1 = (corr_num_disagree_t1correct1 / denom_disagree_t1correct1) if denom_disagree_t1correct1 > 0 else 0
        rate_ind_t1c1 = (ind_num_disagree_t1correct1 / denom_disagree_t1correct1) if denom_disagree_t1correct1 > 0 else 0
        rate_corr_t1c2 = (corr_num_disagree_t1correct2 / denom_disagree_t1correct2) if denom_disagree_t1correct2 > 0 else 0
        rate_ind_t1c2 = (ind_num_disagree_t1correct2 / denom_disagree_t1correct2) if denom_disagree_t1correct2 > 0 else 0

        results_list.append({
            'run_id_timestamp': run_id_ts, 'team_name': team_name, 'run_name': run_name,
            'denom_disagree_overall': denom_disagree_overall,
            'corr_N_disagree_overall': corr_num_disagree_overall,
            'ind_N_disagree_overall': ind_num_disagree_overall,
            'no_maj_F_disagree_overall': no_maj_final_disagree_overall,
            'correction_rate_overall': rate_corr_overall,
            'induction_rate_overall': rate_ind_overall,
            'denom_disagree_t1c1': denom_disagree_t1correct1,
            'corr_N_disagree_t1c1': corr_num_disagree_t1correct1,
            'ind_N_disagree_t1c1': ind_num_disagree_t1correct1,
            'no_maj_F_disagree_t1c1': no_maj_final_disagree_t1correct1,
            'correction_rate_t1c1': rate_corr_t1c1,
            'induction_rate_t1c1': rate_ind_t1c1,
            'denom_disagree_t1c2': denom_disagree_t1correct2,
            'corr_N_disagree_t1c2': corr_num_disagree_t1correct2,
            'ind_N_disagree_t1c2': ind_num_disagree_t1correct2,
            'no_maj_F_disagree_t1c2': no_maj_final_disagree_t1correct2,
            'correction_rate_t1c2': rate_corr_t1c2,
            'induction_rate_t1c2': rate_ind_t1c2,
        })
        
        if args.verbose:
            print(f"  Results for {team_name}/{run_name}:")
            print(f"    Overall Disagree: Denom={denom_disagree_overall}, RateCorrect={rate_corr_overall:.2f}, RateInduct={rate_ind_overall:.2f}")
            print(f"    T1Correct=1 Disagree: Denom={denom_disagree_t1correct1}, RateCorrect={rate_corr_t1c1:.2f}, RateInduct={rate_ind_t1c1:.2f}")
            print(f"    T1Correct=2 Disagree: Denom={denom_disagree_t1correct2}, RateCorrect={rate_corr_t1c2:.2f}, RateInduct={rate_ind_t1c2:.2f}")

    if not results_list:
        print("集計対象となるデータがありませんでした。")
        return

    df_results = pd.DataFrame(results_list)

    try:
        df_results.to_csv(output_rates_csv_file, index=False, encoding="utf-8-sig")
        print(f"\n回答修正率と誤答誘発率の集計結果(初期正解者数別)がCSVファイルに保存されました: {output_rates_csv_file.resolve()}")
    except Exception as e:
        print(f"エラー: 集計結果CSVファイルへの書き込み中にエラーが発生しました: {output_rates_csv_file.resolve()}, 詳細: {e}")

    if not df_results.empty:
        rate_cols_to_agg = [
            'correction_rate_overall', 'induction_rate_overall',
            'correction_rate_t1c1', 'induction_rate_t1c1',
            'correction_rate_t1c2', 'induction_rate_t1c2'
        ]
        # mean と std を計算する列のリストを作成
        agg_functions_for_rates = {col: ['mean', 'std'] for col in rate_cols_to_agg}
        
        # sum を計算する列のリストを作成
        count_cols_to_agg = [
            'denom_disagree_overall', 'corr_N_disagree_overall', 'ind_N_disagree_overall', 'no_maj_F_disagree_overall',
            'denom_disagree_t1c1', 'corr_N_disagree_t1c1', 'ind_N_disagree_t1c1', 'no_maj_F_disagree_t1c1',
            'denom_disagree_t1c2', 'corr_N_disagree_t1c2', 'ind_N_disagree_t1c2', 'no_maj_F_disagree_t1c2'
        ]
        agg_functions_for_counts = {col: ['sum'] for col in count_cols_to_agg}

        agg_dict = {
            **agg_functions_for_rates,
            **agg_functions_for_counts,
            'run_name': 'nunique' # nunique はタプルではなく直接指定
        }
        
        # groupbyキーをインデックスにしないために as_index=False を指定
        team_summary = df_results.groupby(['run_id_timestamp', 'team_name'], as_index=False).agg(agg_dict)

        # MultiIndexヘッダーをフラットにする
        new_cols = []
        for col_top, col_bottom in team_summary.columns:
            if col_bottom == '': # run_id_timestamp, team_name
                new_cols.append(col_top)
            elif col_top == 'run_name' and col_bottom == 'nunique':
                 new_cols.append('num_runs_analyzed')
            elif col_bottom in ['mean', 'std', 'sum']:
                new_cols.append(f"{col_bottom}_{col_top}") # mean_correction_rate_overall のような形に
            else: # これ以外のケースは基本ないはずだが念のため
                new_cols.append(f"{col_top}_{col_bottom}")
        team_summary.columns = new_cols
        
        # 全runでの合計から再計算した率も参考として追加
        for cat in ['overall', 't1c1', 't1c2']:
            total_denom_col = f'sum_denom_disagree_{cat}'
            total_corr_N_col = f'sum_corr_N_disagree_{cat}'
            total_ind_N_col = f'sum_ind_N_disagree_{cat}'
            
            # total_denom_col が存在し、かつ0より大きい場合のみ率を計算
            if total_denom_col in team_summary.columns:
                team_summary[f'overall_corr_rate_from_totals_{cat}'] = team_summary.apply(
                    lambda row: (row[total_corr_N_col] / row[total_denom_col]) if row.get(total_denom_col, 0) > 0 else 0, axis=1
                )
                team_summary[f'overall_ind_rate_from_totals_{cat}'] = team_summary.apply(
                    lambda row: (row[total_ind_N_col] / row[total_denom_col]) if row.get(total_denom_col, 0) > 0 else 0, axis=1
                )
        
        team_summary.fillna(0, inplace=True)

        team_summary_file = run_analysis_path / "team_correction_induction_rates_avg_summary_v2.csv"
        try:
            team_summary.to_csv(team_summary_file, index=False, encoding="utf-8-sig")
            print(f"チームごとの平均回答修正率・誤答誘発率(初期正解者数別)がCSVファイルに保存されました: {team_summary_file.resolve()}")
            if args.verbose:
                print("\nTeam Summary (v2):")
                print(team_summary.head())
        except Exception as e:
            print(f"エラー: チーム集計CSVファイルへの書き込み中にエラーが発生しました: {team_summary_file.resolve()}, 詳細: {e}")
            
    if args.verbose:
        print(f"--- Correction/Induction Rate Calculation (v2) for Run ID: {args.run_id} Complete ---")

if __name__ == "__main__":
    main()