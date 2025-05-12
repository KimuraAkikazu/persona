import os
import csv
import re

BFI_QUESTIONS = [
    "Is talkative",
    "Tends to find fault with others",
    "Does a thorough job",
    "Is depressed, blue",
    "Is original, comes up with new ideas",
    "Is reserved",
    "Is helpful and unselfish with others",
    "Can be somewhat careless",
    "Is relaxed, handles stress well",
    "Is curious about many different things",
    "Is full of energy",
    "Starts quarrels with others",
    "Is a reliable worker",
    "Can be tense",
    "Is ingenious, a deep thinker",
    "Generates a lot of enthusiasm",
    "Has a forgiving nature",
    "Tends to be disorganized",
    "Worries a lot",
    "Has an active imagination",
    "Tends to be quiet",
    "Is generally trusting",
    "Tends to be lazy",
    "Is emotionally stable, not easily upset",
    "Is inventive",
    "Has an assertive personality",
    "Can be cold and aloof",
    "Perseveres until the task is finished",
    "Can be moody",
    "Values artistic, aesthetic experiences",
    "Is sometimes shy, inhibited",
    "Is considerate and kind to almost everyone",
    "Does things efficiently",
    "Remains calm in tense situations",
    "Prefers work that is routine",
    "Is outgoing, sociable",
    "Is sometimes rude to others",
    "Makes plans and follows through with them",
    "Gets nervous easily",
    "Likes to reflect, play with ideas",
    "Has few artistic interests",
    "Likes to cooperate with others",
    "Is easily distracted",
    "Is sophisticated in art, music, or literature",
]

BFI_SCALE = {
    "Extraversion": [1, 6, 11, 16, 21, 26, 31, 36],
    "Agreeableness": [2, 7, 12, 17, 22, 27, 32, 37, 42],
    "Conscientiousness": [3, 8, 13, 18, 23, 28, 33, 38, 43],
    "Neuroticism": [4, 9, 14, 19, 24, 29, 34, 39],
    "Openness": [5, 10, 15, 20, 25, 30, 35, 40, 41, 44],
}

REVERSE_SCORED = [6, 21, 31, 2, 12, 27, 37, 8, 18, 23, 43, 9, 24, 34, 35, 41]
REVERSE_MAPPING = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}


def compute_bfi_scores(all_scores):
    converted = []
    for i, val in enumerate(all_scores, start=1):
        if i in REVERSE_SCORED and val in REVERSE_MAPPING:
            converted.append(REVERSE_MAPPING[val])
        else:
            converted.append(val)
    results = {}
    for trait, items in BFI_SCALE.items():
        results[trait] = sum(converted[item - 1] for item in items)
    return results


def run_bfi_test_with_analyzer(
    persona_agent, analyzer_agent, test_phase, csv_file="bfi_results.csv"
):
    n_questions = len(BFI_QUESTIONS)
    scores = []
    for i, q in enumerate(BFI_QUESTIONS, start=1):
        num = persona_agent.get_bfi_score(q, i, n_questions)
        scores.append(num if 1 <= num <= 5 else 0)
    final = compute_bfi_scores(scores)
    write_header = not os.path.exists(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "AgentName",
                "TestPhase",
                "Extraversion",
                "Agreeableness",
                "Conscientiousness",
                "Neuroticism",
                "Openness",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {"AgentName": persona_agent.name, "TestPhase": test_phase, **final}
        )
    print(
        f"[INFO] BFI test done for {persona_agent.name} ({test_phase}), saved to {csv_file}"
    )
