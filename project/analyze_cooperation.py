#!/usr/bin/env python
"""
imaokre
MMLU Collaboration Analysis Script

This script analyzes discussion logs of LLM agents solving MMLU problems,
with functions for:
1. Language analysis of discussions (turns 2-3) using sentiment analysis and word clouds
2. Tracking correct/incorrect answer transitions between turns
3. Measuring individual agent answer changes from correct to incorrect and vice versa

Usage:
    python analyze_mmlu_collaboration.py --run-id <run_id>

Arguments:
    --run-id       ID of the run to analyze (required)
    --results-dir  Directory containing results data (default: "results")
    --verbose      Show detailed progress logs
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze MMLU agent collaboration')
    parser.add_argument('--run-id', required=True, help='Run ID to analyze')
    parser.add_argument('--results-dir', default='results', help='Results directory')
    parser.add_argument('--verbose', action='store_true', help='Show detailed logs')
    return parser.parse_args()


def load_debate_jsonl(path):
    """Load a debate JSONL file into a pandas DataFrame."""
    records = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        return pd.DataFrame(records)
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred while loading {path}: {e}")
        return pd.DataFrame()


def load_mmlu_answers(path):
    """Load correct MMLU answers from a given file."""
    # In a real implementation, this would load correct answers from mmlu.pkl
    # For now, we'll use a placeholder that returns a sample answer
    # Actual implementation would depend on mmlu.pkl format
    return {"1": "A", "2": "B", "3": "C"}


def analyze_language_features(df, output_dir):
    """
    Analyze language features of turns 2-3 discussions.

    Parameters:
        df (DataFrame): DataFrame containing debate data
        output_dir (Path): Directory to save analysis results
    """
    if df.empty or 'turn' not in df.columns or 'reasoning' not in df.columns or 'agent' not in df.columns:
        print(f"    Skipping language features analysis for {output_dir.name}: DataFrame is empty or missing required columns.")
        return {
            'sentiment': pd.DataFrame(),
            'word_counts': {},
            'agreement': pd.DataFrame()
        }

    # Filter for turns 2 and 3
    turns_df = df[df['turn'].isin([2, 3])].copy()
    if turns_df.empty:
        print(f"    No data for turns 2-3 found for language analysis in {output_dir.name}.")
        # Ensure output directory exists even if no data, so summary can be written later
        output_dir.mkdir(parents=True, exist_ok=True)
        # Create empty placeholder files to indicate analysis was attempted but no data was found for this section
        # This helps distinguish from permissions errors later
        (output_dir / 'agent_sentiment.csv').write_text('')
        (output_dir / 'common_words.json').write_text('{}')
        (output_dir / 'agent_agreement.csv').write_text('')

        return {
            'sentiment_path': str(output_dir / 'agent_sentiment.csv'),
            'common_words_path': str(output_dir / 'common_words.json'),
            'agreement_path': str(output_dir / 'agent_agreement.csv')
        }

    # Ensure output directory exists before writing files
    output_dir.mkdir(parents=True, exist_ok=True)


    # 1. Sentiment Analysis
    turns_df['sentiment'] = turns_df['reasoning'].apply(
        lambda text: TextBlob(str(text)).sentiment.polarity
    )

    agent_sentiment = turns_df.groupby('agent')['sentiment'].mean().reset_index()

    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='agent', y='sentiment', data=agent_sentiment)
        plt.title(f'Average Sentiment by Agent (Turns 2-3) - {output_dir.name}')
        plt.xlabel('Agent')
        plt.ylabel('Sentiment Polarity (-1 to 1)')
        plt.tight_layout()
        plt.savefig(output_dir / 'agent_sentiment.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"    Error generating sentiment plot for {output_dir.name}: {e}")

    agent_sentiment.to_csv(output_dir / 'agent_sentiment.csv', index=False)

    # 2. Word Cloud for each agent
    for agent, group in turns_df.groupby('agent'):
        text = ' '.join(group['reasoning'].fillna(''))
        if text.strip():
            try:
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    stopwords=set(STOPWORDS),
                    max_words=100
                ).generate(text)

                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Word Cloud for {agent} (Turns 2-3) - {output_dir.name}')
                plt.tight_layout()
                plt.savefig(output_dir / f'wordcloud_{agent}.png', dpi=300)
                plt.close()
            except ValueError as e:
                print(f"    Could not generate word cloud for agent {agent} in {output_dir.name}: {e}")
            except Exception as e:
                 print(f"    Error generating word cloud for agent {agent} in {output_dir.name}: {e}")


    # 3. Common phrases/words used by each agent
    word_counts = {}
    for agent, group in turns_df.groupby('agent'):
        text = ' '.join(group['reasoning'].fillna(''))
        words = [w.lower() for w in text.split() if w.lower() not in STOPWORDS and len(w) > 3]
        word_counts[agent] = Counter(words).most_common(10)

    with open(output_dir / 'common_words.json', 'w', encoding='utf-8') as f:
        json.dump(word_counts, f, indent=2)

    # 4. Agreement phrases
    agreement_phrases = ['agree', 'same', 'correct', 'right', 'confident', 'consensus']
    turns_df['agreement_score'] = turns_df['reasoning'].apply(
        lambda text: sum(1 for phrase in agreement_phrases if phrase in str(text).lower())
    )

    agent_agreement = turns_df.groupby('agent')['agreement_score'].mean().reset_index()

    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='agent', y='agreement_score', data=agent_agreement)
        plt.title(f'Agreement Language Score by Agent (Turns 2-3) - {output_dir.name}')
        plt.xlabel('Agent')
        plt.ylabel('Average Agreement Score')
        plt.tight_layout()
        plt.savefig(output_dir / 'agent_agreement.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"    Error generating agreement plot for {output_dir.name}: {e}")


    agent_agreement.to_csv(output_dir / 'agent_agreement.csv', index=False)

    return {
        'sentiment_path': str(output_dir / 'agent_sentiment.csv'),
        'common_words_path': str(output_dir / 'common_words.json'),
        'agreement_path': str(output_dir / 'agent_agreement.csv')
    }


def calculate_answer_transitions(df, correct_answers, output_dir):
    """
    Calculate transitions between correct and incorrect answers across turns.
    """
    if df.empty or not all(col in df.columns for col in ['q_id', 'turn', 'answer']):
        print(f"    Skipping answer transitions for {output_dir.name}: DataFrame is empty or missing required columns.")
        # Ensure output directory exists even if no data
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'answer_transitions.csv').write_text("transition_type,count\nNoData,0\n")
        return {}

    # Ensure output directory exists before writing files
    output_dir.mkdir(parents=True, exist_ok=True)

    df['is_correct'] = df.apply(
        lambda row: row['answer'] == correct_answers.get(str(row['q_id']), None),
        axis=1
    )

    turn_results = []
    # Ensure q_id is treated consistently (e.g., as string if keys in correct_answers are strings)
    df['q_id_str'] = df['q_id'].astype(str)

    for (q_id_str, turn), group in df.groupby(['q_id_str', 'turn']):
        answer_counts = group['answer'].value_counts()
        majority_answer = answer_counts.idxmax() if not answer_counts.empty else None
        is_correct = majority_answer == correct_answers.get(q_id_str, None)
        turn_results.append({
            'q_id': q_id_str,
            'turn': turn,
            'majority_answer': majority_answer,
            'is_correct': is_correct
        })

    turn_df = pd.DataFrame(turn_results)
    if turn_df.empty:
        print(f"    No turn data to process for answer transitions in {output_dir.name}.")
        (output_dir / 'answer_transitions.csv').write_text("transition_type,count\nNoTurnData,0\n")
        return {}

    transitions = {
        'correct_to_incorrect': 0,
        'incorrect_to_correct': 0,
        'stayed_correct': 0,
        'stayed_incorrect': 0
    }

    for q_id_str in turn_df['q_id'].unique():
        q_turns = turn_df[turn_df['q_id'] == q_id_str].sort_values('turn')
        if len(q_turns) < 2: # Need at least two turns to see a transition
             continue
        for i in range(len(q_turns) - 1):
            current = q_turns.iloc[i]
            next_turn = q_turns.iloc[i + 1]
            if current['is_correct'] and not next_turn['is_correct']:
                transitions['correct_to_incorrect'] += 1
            elif not current['is_correct'] and next_turn['is_correct']:
                transitions['incorrect_to_correct'] += 1
            elif current['is_correct'] and next_turn['is_correct']:
                transitions['stayed_correct'] += 1
            else: # not current['is_correct'] and not next_turn['is_correct']
                transitions['stayed_incorrect'] += 1

    labels = list(transitions.keys())
    values = list(transitions.values())

    try:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, color=['red', 'green', 'blue', 'orange'])
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        plt.title(f'Answer Correctness Transitions - {output_dir.name}')
        plt.xlabel('Transition Type')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(output_dir / 'answer_transitions.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"    Error generating transitions plot for {output_dir.name}: {e}")


    with open(output_dir / 'answer_transitions.csv', 'w', newline='') as f:
        f.write("transition_type,count\n")
        for k, v in transitions.items():
            f.write(f"{k},{v}\n")
    return transitions


def analyze_agent_answer_changes(df, correct_answers, output_dir):
    """
    Analyze how often each agent changes from correct to incorrect or vice versa.
    """
    if df.empty or not all(col in df.columns for col in ['q_id', 'turn', 'answer', 'agent']):
        print(f"    Skipping agent answer changes for {output_dir.name}: DataFrame is empty or missing required columns.")
        # Ensure output directory exists even if no data
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'agent_answer_changes.csv').write_text("agent,change_type,count\nNoData,NoData,0\n")
        return {}

    # Ensure output directory exists before writing files
    output_dir.mkdir(parents=True, exist_ok=True)

    df['is_correct'] = df.apply(
        lambda row: row['answer'] == correct_answers.get(str(row['q_id']), None),
        axis=1
    )

    agent_changes = {agent: {'correct_to_incorrect': 0, 'incorrect_to_correct': 0}
                    for agent in df['agent'].unique()}

    # Ensure q_id is treated consistently (e.g., as string)
    df['q_id_str'] = df['q_id'].astype(str)

    for agent in df['agent'].unique():
        agent_df = df[df['agent'] == agent]
        for q_id_str in agent_df['q_id_str'].unique():
            q_turns = agent_df[agent_df['q_id_str'] == q_id_str].sort_values('turn')
            if len(q_turns) <= 1:
                continue
            for i in range(len(q_turns) - 1):
                current = q_turns.iloc[i]
                next_turn = q_turns.iloc[i + 1]
                if current['is_correct'] and not next_turn['is_correct']:
                    agent_changes[agent]['correct_to_incorrect'] += 1
                elif not current['is_correct'] and next_turn['is_correct']:
                    agent_changes[agent]['incorrect_to_correct'] += 1

    changes_list = []
    for agent, changes in agent_changes.items():
        for change_type, count in changes.items():
            changes_list.append({
                'agent': agent,
                'change_type': change_type,
                'count': count
            })

    changes_df = pd.DataFrame(changes_list)
    if changes_df.empty:
        print(f"    No agent answer change data to plot for {output_dir.name}.")
        pivot_df = pd.DataFrame([{'agent': 'NoAgentData', 'correct_to_incorrect': 0, 'incorrect_to_correct': 0}])
    else:
        try:
            plt.figure(figsize=(12, 6))
            sns.barplot(x='agent', y='count', hue='change_type', data=changes_df)
            plt.title(f'Agent Answer Changes Between Turns - {output_dir.name}')
            plt.xlabel('Agent')
            plt.ylabel('Count')
            plt.legend(title='Change Type')
            plt.tight_layout()
            plt.savefig(output_dir / 'agent_answer_changes.png', dpi=300)
            plt.close()
        except Exception as e:
            print(f"    Error generating agent changes plot for {output_dir.name}: {e}")

        pivot_df = changes_df.pivot_table(index='agent', columns='change_type', values='count', fill_value=0).reset_index()

    pivot_df.to_csv(output_dir / 'agent_answer_changes.csv', index=False)

    return agent_changes


def analyze_sub_run(run_id_arg, sub_run_data_dir, base_output_dir_for_parent_team, correct_answers, verbose=False):
    """
    Analyze a single sub-run's data (e.g., data within a 'run00' folder).

    Parameters:
        run_id_arg (str): The original run ID.
        sub_run_data_dir (Path): Path to the sub-run directory containing debate.jsonl (e.g., .../TeamMixed/run00).
        base_output_dir_for_parent_team (Path): Base output directory for the parent team
                                               (e.g., analysis/<run_id>/TeamMixed).
        correct_answers (dict): Dictionary of correct answers.
        verbose (bool): Verbosity flag.
    """
    if verbose:
        print(f"  Analyzing sub-run data in directory: {sub_run_data_dir}")

    debate_path = sub_run_data_dir / "debate.jsonl"
    if not debate_path.exists():
        if verbose:
            print(f"    Skipping - no debate.jsonl found in {sub_run_data_dir}")
        return None

    # Name of the sub-run directory (e.g., "run00")
    sub_run_name = sub_run_data_dir.name

    # Full output directory for this specific sub-run
    # e.g., analysis/<run_id>/<parent_team_name>/<sub_run_name>/
    output_dir_for_sub_run = base_output_dir_for_parent_team / sub_run_name
    # This directory will be created inside analysis functions if data is available,
    # or explicitly created by those functions even if no data for placeholder files.
    # It's also created before the summary file below.
    output_dir_for_sub_run.mkdir(parents=True, exist_ok=True)


    parent_team_name = base_output_dir_for_parent_team.name
    analysis_target_identifier = f"{parent_team_name}_{sub_run_name}"

    if verbose:
        print(f"    Loading data from {debate_path} for {analysis_target_identifier}")

    df = load_debate_jsonl(debate_path)
    if df.empty:
        if verbose:
            print(f"    Skipping {analysis_target_identifier} - debate.jsonl in {sub_run_data_dir} is empty or failed to load.")
        # Create the output directory and an empty summary file to indicate it was processed but had no data
        output_dir_for_sub_run.mkdir(parents=True, exist_ok=True)
        empty_results = {
            'analysis_target_identifier': analysis_target_identifier,
            'parent_team_name': parent_team_name,
            'sub_run_name': sub_run_name,
            'run_id': run_id_arg,
            'data_source_path': str(sub_run_data_dir),
            'output_path': str(output_dir_for_sub_run),
            'status': 'skipped_empty_data',
            'transitions': {},
            'agent_changes': {},
            'language_features_summary': {}
        }
        with open(output_dir_for_sub_run / 'analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump(empty_results, f, indent=2)
        # Return the empty result structure so it's included in the overall summary
        return empty_results


    if verbose:
        print(f"    Analyzing language features for {analysis_target_identifier}...")
    language_features_paths = analyze_language_features(df, output_dir_for_sub_run) # Returns paths now

    if verbose:
        print(f"    Calculating answer transitions for {analysis_target_identifier}...")
    transitions = calculate_answer_transitions(df, correct_answers, output_dir_for_sub_run)

    if verbose:
        print(f"    Analyzing agent answer changes for {analysis_target_identifier}...")
    agent_changes = analyze_agent_answer_changes(df, correct_answers, output_dir_for_sub_run)

    results = {
        'analysis_target_identifier': analysis_target_identifier,
        'parent_team_name': parent_team_name,
        'sub_run_name': sub_run_name,
        'run_id': run_id_arg,
        'data_source_path': str(sub_run_data_dir),
        'output_path': str(output_dir_for_sub_run),
        'status': 'success',
        'transitions': transitions,
        'agent_changes': agent_changes,
        'language_features_summary': language_features_paths
    }

    with open(output_dir_for_sub_run / 'analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"    Analysis complete for {analysis_target_identifier}, results in {output_dir_for_sub_run}")

    return results


def main():
    """Main function to run the analysis."""
    args = parse_args()

    results_dir = Path(args.results_dir)
    run_dir = results_dir / args.run_id # e.g., results/20250519_153843

    # Base output directory for the entire run_id
    # This path is relative to the script's execution directory
    # MODIFIED: Removed hardcoded 'project'
    output_dir_for_run = Path('analysis') / args.run_id
    # The mkdir call here ensures the base directory exists before any sub-directories are created
    # within it by analyze_sub_run.
    output_dir_for_run.mkdir(parents=True, exist_ok=True)


    if args.verbose:
        print(f"Starting analysis for run: {args.run_id}")
        print(f"Results directory: {run_dir}")
        print(f"Output directory for this run: {output_dir_for_run}")

    correct_answers = load_mmlu_answers(None) # Placeholder

    if args.verbose:
        print(f"Loaded correct answers for {len(correct_answers)} questions (placeholder)")

    # Find all parent team folders (e.g., TeamMixed, TeamA) within the run_dir
    if not run_dir.exists():
        print(f"Error: Run directory {run_dir} does not exist.")
        return

    parent_team_folders = [p for p in run_dir.iterdir() if p.is_dir()]

    if args.verbose:
        print(f"Found {len(parent_team_folders)} potential parent team folder(s): {[f.name for f in parent_team_folders]}")

    all_run_results = []
    total_sub_runs_analyzed = 0
    sub_runs_skipped_no_data = 0


    for parent_folder in parent_team_folders: # e.g., results/20250519_153843/TeamMixed
        if args.verbose:
            print(f"\nProcessing parent team folder: {parent_folder.name}")

        # Find all sub-run directories (e.g., run00, run01) within this parent_folder
        sub_run_folders = [p for p in parent_folder.iterdir() if p.is_dir() and p.name.startswith('run')]

        if not sub_run_folders:
            if args.verbose:
                print(f"  No sub-run folders (starting with 'run') found in {parent_folder.name}. Skipping.")
            continue


        if args.verbose:
            print(f"  Found {len(sub_run_folders)} sub-run folder(s) in {parent_folder.name}: {[f.name for f in sub_run_folders]}")

        # Define the base output directory for this parent_folder's sub-runs
        # e.g., analysis/20250519_153843/TeamMixed/
        output_location_for_parent_team = output_dir_for_run / parent_folder.name
        # analyze_sub_run will create output_location_for_parent_team / sub_run_name

        for sub_run_dir in sub_run_folders: # e.g., results/20250519_153843/TeamMixed/run00
            result = analyze_sub_run(
                args.run_id,
                sub_run_dir,    # Actual data directory with debate.jsonl
                output_location_for_parent_team, # Base for this parent's outputs
                correct_answers,
                args.verbose
            )
            # analyze_sub_run now returns a result dictionary even if skipped due to empty data
            # It only returns None if debate.jsonl file was not found
            if result: # Check if result is not None (meaning debate.jsonl existed)
                 all_run_results.append(result)
                 total_sub_runs_analyzed += 1
                 if result.get('status') == 'skipped_empty_data':
                     sub_runs_skipped_no_data += 1
            # If result is None (debate.jsonl not found), it's already printed a verbose message inside analyze_sub_run


    if args.verbose:
        print(f"\nAnalysis complete.")
        print(f"Total sub-runs processed (data found): {total_sub_runs_analyzed}")
        if sub_runs_skipped_no_data > 0:
             print(f"({sub_runs_skipped_no_data} sub-runs skipped due to empty data)")
        print(f"Analyzed across {len(parent_team_folders)} parent team folder(s).")


    if all_run_results:
        # Added a check for an empty list *after* processing, in case no debate.jsonl was found anywhere
        if args.verbose:
             print(f"Consolidating results for {len(all_run_results)} processed sub-run(s).")

        overall_summary_path = output_dir_for_run / f"{args.run_id}_overall_analysis.json"
        try:
            with open(overall_summary_path, 'w', encoding='utf-8') as f:
                json.dump(all_run_results, f, indent=2)
            if args.verbose:
                print(f"Overall summary of all processed sub-runs saved to: {overall_summary_path}")
        except Exception as e:
             print(f"Error saving overall summary to {overall_summary_path}: {e}")

    elif args.verbose:
        print("No sub-runs were processed (no debate.jsonl files found) to create an overall summary.")


if __name__ == "__main__":
    main()