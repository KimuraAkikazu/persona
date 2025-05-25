#!/usr/bin/env python
# answer_change_analyzer.py

import argparse
import pandas as pd
from pathlib import Path
import json

def parse_args():
    """コマンドライン引数を解析します."""
    parser = argparse.ArgumentParser(
        description="Extracts agent answers from debate logs and saves them to a CSV file."
    )
    parser.add_argument(
        "--results-dir",
        default="results",  # projectディレクトリからの相対パス
        help="実験結果が格納されているディレクトリ (default: results, projectディレクトリからの相対パス)",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="分析対象のrun_id (タイムスタンプ付きのフォルダ名)",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis", # projectディレクトリからの相対パス
        help="分析結果のCSVファイルを出力するディレクトリ (default: analysis, projectディレクトリからの相対パス). "
             "実際の出力は output-dir/run-id/ になります。",
    )
    parser.add_argument(
        "--output-filename",
        default="all_answers_log.csv",
        help="出力するCSVファイル名 (default: all_answers_log.csv)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細なログ出力を有効にする"
    )
    return parser.parse_args()

def main():
    """メイン処理."""
    args = parse_args()

    # スクリプトの実行場所が /project であるため、カレントディレクトリを基準とする
    current_dir = Path(".") # Path.cwd() と同じだが、ここでは "." で明示

    base_results_path = current_dir / args.results_dir
    run_id_path = base_results_path / args.run_id

    base_output_path = current_dir / args.output_dir
    run_output_path = base_output_path / args.run_id
    run_output_path.mkdir(parents=True, exist_ok=True)

    output_csv_file = run_output_path / args.output_filename

    if args.verbose:
        print(f"--- Answer Extraction Started for Run ID: {args.run_id} ---")
        print(f"Script current directory assumed: {current_dir.resolve()}")
        print(f"Source Results Path: {run_id_path.resolve()}")
        print(f"Output CSV will be saved to: {output_csv_file.resolve()}")

    if not run_id_path.exists() or not run_id_path.is_dir():
        print(f"エラー: 指定されたrun_idのディレクトリが見つかりません: {run_id_path.resolve()}")
        return

    all_extracted_answers = []
    team_dirs = sorted([d for d in run_id_path.iterdir() if d.is_dir()])

    if not team_dirs:
        print(f"run_idディレクトリ内にチームディレクトリが見つかりません: {run_id_path.resolve()}")
        return

    for team_dir in team_dirs:
        team_name = team_dir.name
        if args.verbose:
            print(f"\nProcessing Team: {team_name}")

        run_xx_dirs = sorted([d for d in team_dir.iterdir() if d.is_dir() and d.name.startswith("run")])

        if not run_xx_dirs:
            if args.verbose:
                print(f"  チーム '{team_name}' 内に 'runXX' サブディレクトリが見つかりません。")
            continue

        for run_subdir in run_xx_dirs:
            run_name = run_subdir.name
            debate_log_file = run_subdir / "debate.jsonl"

            if not debate_log_file.exists():
                if args.verbose:
                    print(f"  Skipping {team_name}/{run_name}: debate.jsonl が見つかりません。")
                continue

            if args.verbose:
                print(f"  Processing {team_name}/{run_name}/debate.jsonl")

            try:
                with open(debate_log_file, "r", encoding="utf-8") as f:
                    for line_number, line in enumerate(f, 1):
                        try:
                            log_entry = json.loads(line)
                            q_id = log_entry.get("q_id")
                            turn = log_entry.get("turn")
                            agent = log_entry.get("agent")
                            answer = log_entry.get("answer")
                            # reasoning = log_entry.get("reasoning") # 将来的に必要であれば

                            if q_id is not None and turn is not None and agent is not None and answer is not None:
                                all_extracted_answers.append({
                                    "run_id_timestamp": args.run_id,
                                    "team_name": team_name,
                                    "run_name": run_name,
                                    "q_id": q_id,
                                    "agent": agent,
                                    "turn": turn,
                                    "answer": answer,
                                    # "reasoning": reasoning,
                                })
                            else:
                                if args.verbose:
                                    print(f"    Warning: Skipping line {line_number} in {debate_log_file} due to missing key fields (q_id, turn, agent, or answer).")

                        except json.JSONDecodeError:
                            if args.verbose:
                                print(f"    Warning: Skipping line {line_number} in {debate_log_file} due to JSON decoding error.")
            except Exception as e:
                print(f"  エラー: {debate_log_file} の処理中にエラーが発生しました: {e}")

    if all_extracted_answers:
        df_answers = pd.DataFrame(all_extracted_answers)
        try:
            df_answers.to_csv(output_csv_file, index=False, encoding="utf-8-sig") # Excelでの文字化け対策にutf-8-sig
            print(f"\n全エージェントの回答ログがCSVファイルに保存されました: {output_csv_file.resolve()}")
            if args.verbose:
                print(f"保存されたレコード数: {len(df_answers)}")
        except Exception as e:
            print(f"エラー: CSVファイルへの書き込み中にエラーが発生しました: {output_csv_file.resolve()}, 詳細: {e}")
    else:
        print("抽出できる回答データがありませんでした。")

    if args.verbose:
        print(f"--- Answer Extraction for Run ID: {args.run_id} Complete ---")

if __name__ == "__main__":
    main()