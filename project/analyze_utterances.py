#!/usr/bin/env python
# analyze_utterances.py

"""
エージェント発言ログ分析スクリプト + WordCloud 自動生成

指定した run_id（results/<run_id>/）配下の各チームを自動検出し、
 1. 発言回数
 2. 総トークン数
 3. 平均感情極性
 4. WordCloud (チーム全体、エージェントごと、ターンごと)
をチームごとに可視化・出力します。

実行例:
  conda activate llama-test
  python analyze_utterances.py \
    --results-dir results \
    --run-id 20250415_171750 \
    --out-dir analysis/utterances
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS

# デフォルト設定
DEFAULT_RESULTS_DIR = "results"
DEFAULT_OUT_DIR = "analysis/utterances"


def parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="エージェント発言ログ分析スクリプト + WordCloud"
    )
    p.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="実験結果ディレクトリ (results/) のパス",
    )
    p.add_argument(
        "--run-id",
        required=True,
        help="対象の run_id (results/<run-id>/ 以下を自動走査)",
    )
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="出力先ディレクトリ")
    p.add_argument("--verbose", action="store_true", help="詳細ログを表示")
    return p.parse_args()


def load_debate_jsonl(path: Path) -> pd.DataFrame:
    """単一の debate.jsonl を DataFrame に読み込む"""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        records.append(json.loads(line))
    return pd.DataFrame(records)


def compute_tokens(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tokens"] = df["reasoning"].str.split().str.len().fillna(0) + df[
        "answer"
    ].str.split().str.len().fillna(0)
    return df


def compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["polarity"] = df["reasoning"].apply(
        lambda t: TextBlob(str(t)).sentiment.polarity
    )
    return df


def plot_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    out_path: Path,
    ylim: tuple = None,
):
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=df, x=x, y=y)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=12)
    if ylim:
        ax.set_ylim(*ylim)
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(
            f"{int(h) if h == int(h) else f'{h:.2f}'}",
            (p.get_x() + p.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_wordcloud(text: str, title: str, out_path: Path):
    """与えられたテキストで WordCloud を生成・保存"""
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


def analyze_team(
    results_dir: Path, run_id: str, team: str, out_base: Path, verbose: bool
):
    debate_path = results_dir / run_id / team / "debate.jsonl"
    if not debate_path.exists():
        if verbose:
            print(f"  [スキップ] {team} に debate.jsonl がありません")
        return

    if verbose:
        print(f"  [解析] チーム={team}")

    df = load_debate_jsonl(debate_path)
    df = compute_tokens(df)
    df = compute_sentiment(df)

    out_dir = out_base / run_id / team
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 発言回数
    utt = df.groupby("agent").size().reset_index(name="utterances")
    plot_bar(
        utt,
        x="agent",
        y="utterances",
        title=f"{team}: 発言回数",
        ylabel="発言回数",
        out_path=out_dir / "utterances.png",
        ylim=(0, utt["utterances"].max() * 1.1),
    )

    # 2. 総トークン数
    toks = df.groupby("agent")["tokens"].sum().reset_index(name="total_tokens")
    plot_bar(
        toks,
        x="agent",
        y="total_tokens",
        title=f"{team}: 総トークン数",
        ylabel="トークン数合計",
        out_path=out_dir / "total_tokens.png",
        ylim=(0, toks["total_tokens"].max() * 1.1),
    )

    # 3. 平均感情極性
    sent = df.groupby("agent")["polarity"].mean().reset_index(name="avg_polarity")
    plot_bar(
        sent,
        x="agent",
        y="avg_polarity",
        title=f"{team}: 平均感情極性",
        ylabel="平均極性",
        out_path=out_dir / "avg_polarity.png",
        ylim=(-1.0, 1.0),
    )

    # 4. WordCloud — チーム全体
    full_text = "\n".join(df["reasoning"].fillna("") + " " + df["answer"].fillna(""))
    generate_wordcloud(
        full_text,
        title=f"{team}: 全体 WordCloud",
        out_path=out_dir / "wordcloud_all.png",
    )

    # 5. WordCloud — エージェント別
    for agent, sub in df.groupby("agent"):
        text = "\n".join(sub["reasoning"].fillna("") + " " + sub["answer"].fillna(""))
        generate_wordcloud(
            text,
            title=f"{team}/{agent} WordCloud",
            out_path=out_dir / f"wordcloud_{agent}.png",
        )

    # 6. WordCloud — ターン別 (例: Round 1 only)
    for turn, sub in df.groupby("turn"):
        text = "\n".join(sub["reasoning"].fillna("") + " " + sub["answer"].fillna(""))
        generate_wordcloud(
            text,
            title=f"{team}: Round{turn} WordCloud",
            out_path=out_dir / f"wordcloud_turn{turn}.png",
        )


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_base = Path(args.out_dir)

    if args.verbose:
        print(f"=== Run ID: {args.run_id} のチーム一覧取得 ===")
    teams = [p.name for p in (results_dir / args.run_id).iterdir() if p.is_dir()]

    for team in teams:
        analyze_team(results_dir, args.run_id, team, out_base, args.verbose)

    print("全チームの解析が完了しました。")


if __name__ == "__main__":
    main()
