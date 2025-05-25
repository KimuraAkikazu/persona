#!/usr/bin/env python
# analysis_gemini.py

import argparse
import pandas as pd
from pathlib import Path
from scipy import stats
import numpy as np
import json

# Global variable to hold parsed arguments for access in helper functions
ARGS = None

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze MMLU accuracy from experiment runs and perform t-tests.")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Base directory where specific run_id folders are located (default: project/results)",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="The specific run_id (timestamped folder name) to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        default="project/analysis",
        help="Base directory to save analysis output (default: project/analysis). Output will be in output-dir/run-id/",
    )
    parser.add_argument(
        "--popmean",
        type=float,
        default=0.25,
        help="Population mean to test against in the one-sample t-test (default: 0.25 for 4-choice MMLU chance level).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    return parser.parse_args()

def calculate_accuracy_from_metrics(metrics_file_path: Path) -> float | None:
    """
    Calculates accuracy from a metrics.csv file.
    The 'correct' column is expected to contain boolean or 0/1 values.
    """
    global ARGS
    if not metrics_file_path.exists():
        if ARGS and ARGS.verbose:
            print(f"    Metrics file not found: {metrics_file_path}")
        return None
    try:
        df = pd.read_csv(metrics_file_path)
        if "correct" not in df.columns:
            if ARGS and ARGS.verbose:
                print(f"    'correct' column not found in {metrics_file_path}")
            return None
        if df.empty:
            if ARGS and ARGS.verbose:
                print(f"    Metrics file is empty: {metrics_file_path}")
            return None
        
        # Ensure 'correct' column is boolean or numeric for mean calculation
        if df['correct'].dtype == 'bool':
            accuracy = df["correct"].astype(int).mean()
        elif pd.api.types.is_numeric_dtype(df['correct']):
            accuracy = df["correct"].mean()
        else:
            # Attempt to convert if it's like 'True'/'False' strings - robustly
            try:
                # Handle potential string representations of booleans
                df['correct'] = df['correct'].apply(lambda x: str(x).lower() == 'true' if isinstance(x, str) else x)
                accuracy = df["correct"].astype(int).mean()
            except Exception:
                if ARGS and ARGS.verbose:
                    print(f"    Could not convert 'correct' column to numeric in {metrics_file_path}. Values: {df['correct'].unique()[:5]}")
                return None
        return accuracy
    except pd.errors.EmptyDataError:
        if ARGS and ARGS.verbose:
            print(f"    Metrics file is empty or invalid (EmptyDataError): {metrics_file_path}")
        return None
    except Exception as e:
        if ARGS and ARGS.verbose:
            print(f"    Error reading or processing metrics file {metrics_file_path}: {e}")
        return None

def main():
    """Main function to orchestrate the analysis."""
    global ARGS
    ARGS = parse_args()

    base_results_path = Path(ARGS.results_dir)
    run_id_path = base_results_path / ARGS.run_id

    base_output_path = Path(ARGS.output_dir)
    run_output_path = base_output_path / ARGS.run_id
    run_output_path.mkdir(parents=True, exist_ok=True)

    if ARGS.verbose:
        print(f"--- Starting Analysis for Run ID: {ARGS.run_id} ---")
        print(f"Source Results Path: {run_id_path}")
        print(f"Output Analysis Path: {run_output_path}")
        print(f"T-test population mean: {ARGS.popmean}")

    if not run_id_path.exists() or not run_id_path.is_dir():
        print(f"Error: Run ID directory not found: {run_id_path}")
        return

    team_dirs = [d for d in run_id_path.iterdir() if d.is_dir()]

    if not team_dirs:
        print(f"No team directories found in {run_id_path}")
        return

    overall_summary_for_run_id = []

    for team_dir in sorted(team_dirs): # Sort for consistent output order
        team_name = team_dir.name
        if ARGS.verbose:
            print(f"\nProcessing Team: {team_name}")

        team_run_accuracies = []
        run_xx_dirs = sorted([d for d in team_dir.iterdir() if d.is_dir() and d.name.startswith("run")])

        if not run_xx_dirs:
            if ARGS.verbose:
                print(f"  No 'runXX' sub-directories found in {team_dir}")
            continue

        for run_subdir in run_xx_dirs:
            metrics_file = run_subdir / "metrics.csv"
            accuracy = calculate_accuracy_from_metrics(metrics_file)
            if accuracy is not None:
                team_run_accuracies.append(accuracy)
                if ARGS.verbose:
                    print(f"  Run {run_subdir.name}: Accuracy = {accuracy:.4f}")
            else:
                if ARGS.verbose:
                    print(f"  Run {run_subdir.name}: Could not calculate accuracy (metrics file missing, empty, or format error).")
        
        team_analysis_result = {"team_name": team_name, "num_total_runs_in_folder": len(run_xx_dirs) ,"num_runs_with_accuracy": len(team_run_accuracies)}

        if team_run_accuracies:
            accuracies_np_array = np.array(team_run_accuracies)
            mean_accuracy = float(np.mean(accuracies_np_array)) # Ensure JSON serializable
            std_dev_accuracy = float(np.std(accuracies_np_array, ddof=1 if len(accuracies_np_array) > 1 else 0)) # Sample std dev

            team_analysis_result["run_accuracies"] = [float(acc) for acc in team_run_accuracies] # Ensure JSON serializable
            team_analysis_result["mean_accuracy"] = mean_accuracy
            team_analysis_result["std_dev_accuracy"] = std_dev_accuracy
            
            print(f"  Team {team_name}: Mean Accuracy = {mean_accuracy:.4f}, Std Dev = {std_dev_accuracy:.4f} (from {len(team_run_accuracies)} successful runs)")

            if len(team_run_accuracies) > 1:
                # Perform one-sample t-test
                # Filter out NaNs just in case, although calculate_accuracy_from_metrics should prevent them
                valid_accuracies = accuracies_np_array[~np.isnan(accuracies_np_array)]
                if len(valid_accuracies) > 1:
                    t_statistic, p_value = stats.ttest_1samp(valid_accuracies, ARGS.popmean)
                    team_analysis_result["t_statistic"] = float(t_statistic) # Ensure JSON serializable
                    team_analysis_result["p_value"] = float(p_value) # Ensure JSON serializable
                    team_analysis_result["t_test_popmean"] = ARGS.popmean
                    print(f"  One-sample t-test (vs popmean={ARGS.popmean:.2f}): t-statistic = {t_statistic:.4f}, p-value = {p_value:.4f}")
                    if p_value < 0.05:
                        print(f"    Conclusion: The mean accuracy is statistically significantly different from {ARGS.popmean:.2f}.")
                    else:
                        print(f"    Conclusion: The mean accuracy is not statistically significantly different from {ARGS.popmean:.2f}.")
                else:
                    team_analysis_result["t_test_skipped_reason"] = "Not enough valid data points for t-test after NaN removal (<=1)"
                    print(f"  Skipping t-test for team {team_name}: Not enough valid data points ({len(valid_accuracies)}) after potential NaN removal.")
            else:
                team_analysis_result["t_test_skipped_reason"] = "Not enough data points for t-test (<=1 run with accuracy)"
                print(f"  Skipping t-test for team {team_name}: not enough runs with accuracy ({len(team_run_accuracies)}).")
        else:
            team_analysis_result["run_accuracies"] = []
            team_analysis_result["mean_accuracy"] = None
            team_analysis_result["std_dev_accuracy"] = None
            team_analysis_result["t_test_skipped_reason"] = "No runs with calculable accuracy found"
            print(f"  No runs with calculable accuracy found for team {team_name}.")

        team_output_path = run_output_path / team_name
        team_output_path.mkdir(parents=True, exist_ok=True)
        team_summary_filename = team_output_path / "accuracy_analysis_summary.json"
        try:
            with open(team_summary_filename, "w", encoding="utf-8") as f:
                json.dump(team_analysis_result, f, indent=4, ensure_ascii=False)
            if ARGS.verbose:
                print(f"  Saved team analysis summary to: {team_summary_filename}")
        except Exception as e:
            print(f"  Error saving team summary for {team_name} to {team_summary_filename}: {e}")
        
        overall_summary_for_run_id.append(team_analysis_result)

    overall_run_id_summary_filename = run_output_path / f"{ARGS.run_id}_teams_accuracy_summary.json"
    try:
        with open(overall_run_id_summary_filename, "w", encoding="utf-8") as f:
            json.dump(overall_summary_for_run_id, f, indent=4, ensure_ascii=False)
        print(f"\nOverall analysis summary for Run ID '{ARGS.run_id}' saved to: {overall_run_id_summary_filename}")
    except Exception as e:
        print(f"Error saving overall run ID summary to {overall_run_id_summary_filename}: {e}")
        
    print(f"--- Analysis for Run ID: {ARGS.run_id} Complete ---")

if __name__ == "__main__":
    main()