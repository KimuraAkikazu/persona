#!/usr/bin/env python
# project/analyze_runs.py
"""
複数 run を束ねた集計スクリプト
--------------------------------
results/<run_id>/<TeamX>/run00/, run01/, ... という構造を想定し，
  • BFI 箱ひげ図 (trait×score)
  • Accuracy / 平均 total_tokens の run 別集計
を analysis/<run_id>/<TeamX>/ 以下に保存する。
"""

import json
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------- 引数 ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--run-id", required=True, help="対象 run_id (=上位フォルダ名)")
    p.add_argument("--out-dir", default="analysis")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ---------- ヘルパ ----------
def load_bfi_json(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    recs = []
    for agent, scores in data.items():
        for trait, val in scores.items():
            recs.append({"agent": agent, "trait": trait, "score": val})
    return pd.DataFrame(recs)


def main():
    args = parse_args()
    root = Path(args.results_dir) / args.run_id
    assert root.exists(), f"{root} not found"

    for team_dir in [p for p in root.iterdir() if p.is_dir()]:
        team = team_dir.name
        runs = sorted([p for p in team_dir.iterdir() if p.is_dir()])

        if args.verbose:
            print(f"[{team}] detected runs:", ",".join(r.name for r in runs))

        all_bfi, acc_rows = [], []

        for r in runs:
            bfi_path = r / "bfi_scores.json"
            met_path = r / "metrics.csv"
            if not bfi_path.exists() or not met_path.exists():
                print(f"  [warn] skip {r.name} (missing files)")
                continue

            # ---- BFI ----
            df_bfi = load_bfi_json(bfi_path)
            df_bfi["run"] = r.name
            all_bfi.append(df_bfi)

            # ---- metrics ----
            met = pd.read_csv(met_path)
            acc_rows.append(
                {
                    "run": r.name,
                    "accuracy": met["correct"].mean(),
                    "avg_total_tokens": met["total_tokens"].mean(),
                }
            )

        if not all_bfi:
            continue

        df_bfi_all = pd.concat(all_bfi, ignore_index=True)
        df_acc = pd.DataFrame(acc_rows)

        # ---------- 出力先 ----------
        out_base = Path(args.out_dir) / args.run_id / team
        out_base.mkdir(parents=True, exist_ok=True)

        # ======== ★ NEW: run 全体の統計値を計算 ========
        overall = {
            "run": "MEAN",
            "accuracy": df_acc["accuracy"].mean(),
            "avg_total_tokens": df_acc["avg_total_tokens"].mean(),
        }
        var_row = {
            "run": "VAR",
            "accuracy": df_acc["accuracy"].var(ddof=0),
            "avg_total_tokens": df_acc["avg_total_tokens"].var(ddof=0),
        }
        std_row = {
            "run": "STD",
            "accuracy": df_acc["accuracy"].std(ddof=0),
            "avg_total_tokens": df_acc["avg_total_tokens"].std(ddof=0),
        }
        # =============================================
 

        # ---------- 図表 ----------
        # 1) BFI 箱ひげ図
        # 1) BFI 箱ひげ図 (エージェントごと)
        for ag, sub in df_bfi_all.groupby("agent"):
            plt.figure(figsize=(8, 5))
            sns.boxplot(
                data=sub,
                x="trait",
                y="score",
                palette="Set3",
                showfliers=False,
            )
            plt.title(f"{team}/{ag}: Big-Five score distribution")
            plt.ylabel("Score")
            plt.ylim(0, 50)
            plt.tight_layout()
            plt.savefig(out_base / f"bfi_boxplot_{ag}.png", dpi=300)
            plt.close()

        # 2) run 別 Accuracy / token bar
        melt = df_acc.melt(id_vars="run", value_vars=["accuracy", "avg_total_tokens"])
        g = sns.catplot(
            data=melt,
            x="run",
            y="value",
            hue="variable",
            kind="bar",
            height=5,
            aspect=2,
        )
        g.set_axis_labels("", "value")
        g.fig.suptitle(f"{team}: run-level Accuracy & avg tokens", y=1.02)
        plt.tight_layout()
        g.savefig(out_base / "run_metrics_bar.png", dpi=300)
        plt.close()

        # ---------- CSV 保存 ----------
        # 個別 run 行 ＋ 統計 3 行
        df_full = pd.concat(
            [df_acc, pd.DataFrame([overall, var_row, std_row])], ignore_index=True
        )
        df_full.to_csv(out_base / "run_metrics_summary.csv", index=False)

        # JSON でも保存（mean / var / std のみ）
        (out_base / "run_metrics_overall.json").write_text(
            json.dumps(
                {
                    "accuracy": {
                        "mean": overall["accuracy"],
                        "variance": var_row["accuracy"],
                        "std": std_row["accuracy"],
                    },
                    "avg_total_tokens": {
                        "mean": overall["avg_total_tokens"],
                        "variance": var_row["avg_total_tokens"],
                        "std": std_row["avg_total_tokens"],
                    },
                    "n_runs": len(df_acc),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print("Finished aggregation.")


if __name__ == "__main__":
    main()
