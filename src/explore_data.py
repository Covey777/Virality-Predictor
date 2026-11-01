"""
explore_data.py
---------------
Phase 1 EDA: (1) sanity-check the dataset, (2) draw plots that show the patterns we built in.

How to run (from the project root):
    python -m src.explore_data
"""
import json

import matplotlib
matplotlib.use("Agg")  # save figures to files, no pop-up window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C


def sanity_checks(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("1. SANITY CHECKS")
    print("=" * 60)
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Duplicate post_ids: {int(df['post_id'].duplicated().sum())}")

    checks = {
        "views >= 500": (df.views >= 500).all(),
        "likes/comments/shares >= 0": ((df[["likes", "comments", "shares"]] >= 0).all()).all(),
        "engagement rate in (0, 100]": df.engagement_rate.between(0, 100).all(),
        "post_hour in 0..23": df.post_hour.between(0, 23).all(),
        "avg_watch_pct in 0..100": df.avg_watch_pct.between(0, 100).all(),
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print("\nNumeric summary:")
    cols = ["duration_sec", "caption_length", "hashtag_count", "followers", "views", "engagement_rate"]
    print(df[cols].describe().round(2).T[["mean", "std", "min", "50%", "max"]])

    print("\nClass balance (performance_label):")
    print(df.performance_label.value_counts(normalize=True).round(3))


def pattern_report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("2. DO THE PATTERNS LOOK REALISTIC?  (mean engagement % by group)")
    print("=" * 60)
    print("\nBy trending audio:")
    print(df.groupby("trending_audio").engagement_rate.mean().round(2).to_string())
    print("\nBy category:")
    print(df.groupby("category").engagement_rate.mean().sort_values(ascending=False).round(2).to_string())
    print("\nCorrelation with engagement_rate (numeric pre-publish features):")
    num = ["duration_sec", "caption_length", "hashtag_count", "post_hour",
           "trending_audio", "followers", "prev_avg_engagement"]
    print(df[num + ["engagement_rate"]].corr()["engagement_rate"].drop("engagement_rate").round(3).to_string())


def leakage_note(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("3. DATA-LEAKAGE WARNING (important for your viva!)")
    print("=" * 60)
    corr = df[["likes", "comments", "shares"]].sum(axis=1).div(df.views).mul(100).corr(df.engagement_rate)
    print(f"engagement_rate is CALCULATED from likes+comments+shares+views (corr = {corr:.3f}).")
    print("If we fed those columns into the model, it would just re-do the formula -> fake 100% accuracy.")
    print(f"\nSAFE inputs (known before posting): {C.PRE_PUBLISH_FEATURES}")
    print(f"NOT model inputs (known after posting): {C.POST_PUBLISH_METRICS}")
    print("-> Post-publish metrics are used for the target and for audience clustering instead.")


def make_figures(df: pd.DataFrame) -> None:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.THRESHOLDS_PATH) as f:
        th = json.load(f)
    palette = {"Low": "#f87171", "Medium": "#fbbf24", "High": "#34d399"}

    # Fig 1: engagement distribution with class cut-offs
    plt.figure(figsize=(8, 4.5))
    plt.hist(df.engagement_rate, bins=50, color="#8b5cf6", alpha=0.85)
    for v in th.values():
        plt.axvline(v, color="black", linestyle="--")
    plt.title("Engagement rate distribution (dashed = Low/Medium/High cut-offs)")
    plt.xlabel("Engagement rate (%)"); plt.ylabel("Number of posts")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "01_engagement_distribution.png", dpi=130); plt.close()

    # Fig 2: 2x2 pattern panel
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    df.groupby("category").engagement_rate.mean().sort_values().plot.barh(ax=ax[0, 0], color="#ec4899")
    ax[0, 0].set_title("Avg engagement by category"); ax[0, 0].set_xlabel("%")

    dur_bins = pd.cut(df.duration_sec, [0, 10, 20, 30, 45, 60, 90])
    df.groupby(dur_bins, observed=True).engagement_rate.mean().plot(ax=ax[0, 1], marker="o", color="#8b5cf6")
    ax[0, 1].set_title("Avg engagement by video duration (sec)"); ax[0, 1].set_ylabel("%")

    df.groupby("post_hour").engagement_rate.mean().plot(ax=ax[1, 0], marker="o", color="#f59e0b")
    ax[1, 0].set_title("Avg engagement by posting hour"); ax[1, 0].set_ylabel("%")

    df.groupby("hashtag_count").engagement_rate.mean().loc[:16].plot(ax=ax[1, 1], marker="o", color="#10b981")
    ax[1, 1].set_title("Avg engagement by hashtag count"); ax[1, 1].set_ylabel("%")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "02_patterns.png", dpi=130); plt.close()

    # Fig 3: audience behaviour ratios (preview of Phase 3 clustering)
    ratios = pd.DataFrame({
        "like_rate": df.likes / df.views * 100,
        "comment_rate": df.comments / df.views * 100,
        "share_rate": df.shares / df.views * 100,
        "avg_watch_pct": df.avg_watch_pct,
    })
    plt.figure(figsize=(7, 5.5))
    plt.scatter(ratios.comment_rate, ratios.share_rate, c=ratios.avg_watch_pct, cmap="plasma", s=8, alpha=0.6)
    plt.colorbar(label="avg watch %")
    plt.xlabel("Comment rate (% of views)"); plt.ylabel("Share rate (% of views)")
    plt.title("Do audience behaviour groups exist? (preview for K-Means)")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "03_audience_behaviour_preview.png", dpi=130); plt.close()

    # Fig 4: correlation heatmap
    num = ["duration_sec", "caption_length", "hashtag_count", "post_hour", "trending_audio",
           "followers", "prev_avg_engagement", "engagement_rate"]
    corr = df[num].assign(followers=np.log10(df.followers)).corr()
    plt.figure(figsize=(7.5, 6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(num)), num, rotation=45, ha="right"); plt.yticks(range(len(num)), num)
    for i in range(len(num)):
        for j in range(len(num)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(); plt.title("Correlation heatmap (followers on log scale)")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "04_correlation_heatmap.png", dpi=130); plt.close()

    print(f"\nFigures saved to {C.FIG_DIR}")


def main():
    df = pd.read_csv(C.DATASET_PATH)
    sanity_checks(df)
    pattern_report(df)
    leakage_note(df)
    make_figures(df)


if __name__ == "__main__":
    main()
