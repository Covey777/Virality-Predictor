"""
generate_dataset.py
-------------------
Creates a CLEARLY-LABELLED SYNTHETIC dataset of short-video posts.

Why synthetic?
  Real per-post data (views, likes, shares, watch time, posting hour, trending audio...)
  is not publicly available for Instagram, and scraping it breaks platform rules.
  So we simulate it from simple, explainable rules. The rules are ASSUMPTIONS taken from
  common social-media-marketing advice - they are NOT Instagram's real algorithm.

How to run (from the project root):
    python -m src.generate_dataset
"""
import json

import numpy as np
import pandas as pd

from src import config as C


# ------------------------------------------------------------------
# Small helper: a smooth "sweet spot" curve. Returns 1 at `centre`, falls towards 0 far away.
# Used so that e.g. 22-second videos do best, and both 5s and 90s videos do worse.
# ------------------------------------------------------------------
def sweet_spot(x, centre, width):
    return np.exp(-((x - centre) / width) ** 2)


def generate(n=C.N_POSTS, seed=C.RANDOM_SEED, return_latent=False):
    rng = np.random.default_rng(seed)

    # ---------------- 1. Creator & content features (known BEFORE posting) ----------------
    category = rng.choice(C.CATEGORIES, size=n, p=[v[2] for v in C.CATEGORY_PROFILE.values()])
    base_er = np.array([C.CATEGORY_PROFILE[c][0] for c in category])
    typical_dur = np.array([C.CATEGORY_PROFILE[c][1] for c in category])

    followers = np.clip(rng.lognormal(mean=9.5, sigma=1.2, size=n), 500, 5_000_000).astype(int)
    creator_quality = rng.normal(0, 1, n)  # hidden "how good is this creator" factor

    # Bigger accounts usually have LOWER engagement rate; good creators have higher.
    prev_avg_engagement = np.clip(
        5.5 + 1.6 * creator_quality - 0.6 * (np.log10(followers) - 4) + rng.normal(0, 0.4, n),
        0.5, 20,
    ).round(2)

    duration_sec = np.clip(rng.gamma(shape=4, scale=typical_dur / 4), 5, 90).round().astype(int)
    caption_length = np.clip(rng.gamma(shape=2.0, scale=45, size=n), 0, 500).astype(int)
    hashtag_count = np.clip(rng.poisson(6, n), 0, 30)

    # Posting hour: two daily peaks (lunch ~13h and evening ~20h)
    hours = np.arange(24)
    hour_w = 1 + 3 * sweet_spot(hours, 13, 2.5) + 4 * sweet_spot(hours, 20, 2.5)
    post_hour = rng.choice(hours, size=n, p=hour_w / hour_w.sum())
    day_idx = rng.choice(7, size=n, p=[0.13, 0.13, 0.14, 0.14, 0.15, 0.16, 0.15])
    post_day = np.array(C.DAYS)[day_idx]

    trending_audio = (rng.random(n) < 0.30).astype(int)

    # ---------------- 2. Hidden engagement-rate rule (the "ground truth" of our simulation) ---------
    er = (
        0.55 * base_er + 0.55 * prev_avg_engagement                   # category + creator history
        + 1.2 * sweet_spot(duration_sec, 22, 18) - 0.6                # short-ish videos win
        + 1.0 * sweet_spot(hashtag_count, 6, 4) - 0.5                 # 3-9 hashtags is ideal
        + 0.6 * sweet_spot(caption_length, 80, 70) - 0.3              # medium captions
        + 0.9 * ((post_hour >= 18) & (post_hour <= 22))               # evening boost
        + 0.5 * ((post_hour >= 11) & (post_hour <= 14))               # lunch boost
        - 0.7 * (post_hour <= 5)                                      # late-night penalty
        + 0.5 * np.isin(day_idx, [5, 6])                              # weekend boost
        + 1.4 * trending_audio                                        # trending audio boost
        + rng.normal(0, 0.9, n)                                       # random noise (real life is messy)
    )
    er = np.clip(er, 0.3, 30)

    # ---------------- 3. Reach (views) ----------------
    views = (
        followers * np.exp(rng.normal(-0.3, 0.9, n))
        * (1 + 0.6 * trending_audio)
        * np.exp(0.04 * (er - 6))          # better content is shown to slightly more people
    )
    views = np.maximum(views, 500).astype(int)

    # ---------------- 4. Audience behaviour type -> like / comment / share split ----------------
    type_names = list(C.AUDIENCE_TYPES.keys())
    type_p = [v[2] for v in C.AUDIENCE_TYPES.values()]
    a_type = rng.choice(len(type_names), size=n, p=type_p)

    mix = np.array([C.AUDIENCE_TYPES[t][0] for t in type_names])[a_type]       # (n, 3)
    watch_mean = np.array([C.AUDIENCE_TYPES[t][1] for t in type_names])[a_type]

    split = rng.gamma(shape=mix * 40)                     # noisy version of the type's mix
    split = split / split.sum(axis=1, keepdims=True)

    total_interactions = views * er / 100
    likes = np.round(total_interactions * split[:, 0]).astype(int)
    comments = np.round(total_interactions * split[:, 1]).astype(int)
    shares = np.round(total_interactions * split[:, 2]).astype(int)

    # Watch-through %: depends on audience type, and shorter videos are watched more fully
    avg_watch_pct = np.clip(watch_mean - 0.25 * (duration_sec - 25) + rng.normal(0, 7, n), 5, 100).round(1)

    # ---------------- 5. Assemble DataFrame ----------------
    df = pd.DataFrame({
        "post_id": [f"P{i:05d}" for i in range(1, n + 1)],
        "category": category,
        "duration_sec": duration_sec,
        "caption_length": caption_length,
        "hashtag_count": hashtag_count,
        "post_hour": post_hour,
        "post_day": post_day,
        "trending_audio": trending_audio,
        "followers": followers,
        "prev_avg_engagement": prev_avg_engagement,
        "views": views,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "avg_watch_pct": avg_watch_pct,
    })

    # ---------------- 6. Targets ----------------
    # Engagement rate (%) = (likes + comments + shares) / views * 100
    df["engagement_rate"] = ((df.likes + df.comments + df.shares) / df.views * 100).round(3)

    # Low / Medium / High = bottom third / middle third / top third of engagement rate.
    q1, q2 = df["engagement_rate"].quantile([1 / 3, 2 / 3])
    df["performance_label"] = pd.cut(
        df["engagement_rate"], bins=[-np.inf, q1, q2, np.inf], labels=C.LABELS
    ).astype(str)

    thresholds = {"low_medium_cutoff": float(q1), "medium_high_cutoff": float(q2)}
    if return_latent:
        # Ground-truth audience type of each post. NEVER saved to the CSV and NEVER used for training;
        # Phase 3 uses it only to check how well unsupervised K-Means rediscovered the hidden groups.
        return df, thresholds, np.array(type_names)[a_type]
    return df, thresholds


def main():
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df, thresholds = generate()
    df.to_csv(C.DATASET_PATH, index=False)
    with open(C.THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"Saved {len(df):,} synthetic posts -> {C.DATASET_PATH}")
    print(f"Label cut-offs (engagement %): {thresholds}")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nClass balance:")
    print(df["performance_label"].value_counts())


if __name__ == "__main__":
    main()