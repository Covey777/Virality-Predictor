"""
segmentation.py
---------------
Phase 3: UNSUPERVISED learning.
  * K-Means groups posts by HOW their audience behaved (no labels are given to the algorithm).
  * PCA squeezes the behaviour features down to 2 numbers per post so we can draw an "audience map".

Pipeline:  raw metrics -> behaviour features -> StandardScaler -> K-Means -> PCA (for the plot)

Viva notes:
  * K-Means measures DISTANCE, so features must be on the same scale -> StandardScaler.
  * We choose K with two tools: the ELBOW (inertia) and the SILHOUETTE score.
  * PCA is used ONLY for visualisation. K-Means runs on the full scaled features, not on the 2 PCs.
  * Segments are named by looking at each cluster's average behaviour (see name_clusters).

How to run (from the project root):
    python -m src.segmentation
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config as C

SEED = C.RANDOM_SEED

# Plain-English meaning of each segment (used by the dashboard in Phase 4)
SEGMENT_INFO = {
    "Casual Scrollers": {
        "colour": "#94a3b8",
        "description": "Watch briefly and leave a quick like. Wide reach, shallow involvement.",
        "tip": "Hook viewers in the first 2 seconds and add a clear call-to-action to deepen involvement.",
    },
    "Conversation Starters": {
        "colour": "#38bdf8",
        "description": "Unusually talkative audience: many comments compared with likes and shares.",
        "tip": "Ask questions or run polls in the caption, and reply to comments early to keep the thread alive.",
    },
    "Viral Sharers": {
        "colour": "#f472b6",
        "description": "Audience that forwards the post to friends. Strong spread beyond your followers.",
        "tip": "Make content relatable or useful enough to send to a friend, and add 'share this with...' prompts.",
    },
    "Loyal Fans": {
        "colour": "#a78bfa",
        "description": "Watch most of the video and engage steadily. Deep attention, dependable support.",
        "tip": "Keep serving this audience with series-style content; they respond to consistency.",
    },
}


# ======================================================================
# Feature engineering (shared by training AND the Streamlit app)
# ======================================================================
def behaviour_features(likes, comments, shares, avg_watch_pct) -> pd.DataFrame:
    """Works on Series/arrays (whole dataset) or single numbers (one post in the app)."""
    likes, comments, shares = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (likes, comments, shares))
    watch = np.atleast_1d(np.asarray(avg_watch_pct, dtype=float))
    total = np.maximum(likes + comments + shares, 1)          # avoid dividing by zero
    return pd.DataFrame({
        "comment_share": comments / total * 100,
        "share_share": shares / total * 100,
        "avg_watch_pct": watch,
    })[C.CLUSTER_FEATURES]


# ======================================================================
# Choosing K
# ======================================================================
def evaluate_k(X):
    rows = []
    for k in C.K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    res = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(res.k, res.inertia, "o-", color="#8b5cf6")
    ax[0].axvline(C.K_CLUSTERS, color="grey", linestyle="--")
    ax[0].set_title("Elbow method: inertia vs K"); ax[0].set_xlabel("K (number of clusters)")
    ax[0].set_ylabel("Inertia (within-cluster spread)")
    ax[1].plot(res.k, res.silhouette, "o-", color="#ec4899")
    ax[1].axvline(C.K_CLUSTERS, color="grey", linestyle="--")
    ax[1].set_title("Silhouette score vs K (higher = better separated)"); ax[1].set_xlabel("K")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "08_choosing_k.png", dpi=130); plt.close()
    return res


# ======================================================================
# Naming clusters from their centroids
# ======================================================================
def name_clusters(centroids: pd.DataFrame) -> dict:
    """centroids: one row per cluster, in ORIGINAL units. Returns {cluster_id: segment name}."""
    remaining = set(centroids.index)
    names = {}
    for segment, feature in [("Viral Sharers", "share_share"),
                             ("Conversation Starters", "comment_share"),
                             ("Loyal Fans", "avg_watch_pct")]:
        winner = centroids.loc[list(remaining), feature].idxmax()
        names[int(winner)] = segment
        remaining.remove(winner)
    for leftover in remaining:
        names[int(leftover)] = "Casual Scrollers"
    return names


# ======================================================================
# Plots
# ======================================================================
def plot_audience_map(coords, seg_names, centroid_xy, var):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for seg, info in SEGMENT_INFO.items():
        m = seg_names == seg
        ax.scatter(coords[m, 0], coords[m, 1], s=9, alpha=0.5, color=info["colour"], label=seg)
    for seg, (x, y) in centroid_xy.items():
        ax.scatter(x, y, s=180, marker="X", color="black", zorder=5)
        ax.annotate(seg, (x, y), textcoords="offset points", xytext=(8, 8), fontsize=9, weight="bold")
    ax.set_xlabel(f"PC1 ({var[0]:.0%} of variance)"); ax.set_ylabel(f"PC2 ({var[1]:.0%} of variance)")
    ax.set_title("Audience map: K-Means segments projected to 2-D with PCA")
    ax.legend(markerscale=2)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "09_audience_map.png", dpi=130); plt.close()


def plot_profiles(profile: pd.DataFrame):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    titles = {"comment_share": "Comments (% of interactions)", "share_share": "Shares (% of interactions)",
              "avg_watch_pct": "Average watch-through (%)"}
    for a, f in zip(ax, C.CLUSTER_FEATURES):
        a.bar(profile.index, profile[f], color=[SEGMENT_INFO[s]["colour"] for s in profile.index])
        a.set_title(titles[f], fontsize=10); a.tick_params(axis="x", rotation=25)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "10_segment_profiles.png", dpi=130); plt.close()


# ======================================================================
# Main training routine
# ======================================================================
def main():
    C.MODELS_DIR.mkdir(exist_ok=True); C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.DATASET_PATH)
    F = behaviour_features(df.likes, df.comments, df.shares, df.avg_watch_pct)

    scaler = StandardScaler()
    X = scaler.fit_transform(F)

    print("=" * 64 + "\nCHOOSING K\n" + "=" * 64)
    print(evaluate_k(X).round(3).to_string(index=False))
    print(f"-> Using K = {C.K_CLUSTERS} (see reports/figures/08_choosing_k.png)")

    km = KMeans(n_clusters=C.K_CLUSTERS, n_init=10, random_state=SEED).fit(X)
    pca = PCA(n_components=2, random_state=SEED).fit(X)
    coords = pca.transform(X)
    var = pca.explained_variance_ratio_

    centroids = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=C.CLUSTER_FEATURES)
    id_to_name = name_clusters(centroids)
    seg_names = np.array([id_to_name[c] for c in km.labels_])

    profile = centroids.rename(index=id_to_name)
    profile["share_of_posts_%"] = pd.Series(km.labels_).value_counts(normalize=True).mul(100).rename(index=id_to_name)
    profile["avg_engagement_%"] = df.groupby(seg_names).engagement_rate.mean()
    profile["avg_views"] = df.groupby(seg_names).views.mean()
    profile = profile.loc[list(SEGMENT_INFO.keys())]

    print("\n" + "=" * 64 + "\nSEGMENT PROFILES (cluster centres in original units)\n" + "=" * 64)
    print(profile.round(1).to_string())
    print(f"\nSilhouette score (K={C.K_CLUSTERS}): {silhouette_score(X, km.labels_):.3f}")
    print(f"PCA explained variance: PC1 {var[0]:.1%} + PC2 {var[1]:.1%} = {var.sum():.1%}")
    print("PCA loadings (how each PC mixes the original features):")
    print(pd.DataFrame(pca.components_, columns=C.CLUSTER_FEATURES, index=["PC1", "PC2"]).round(2).to_string())

    # ---- Optional synthetic-only validation: did K-Means rediscover the hidden types? ----
    try:
        from src.generate_dataset import generate
        _, _, latent = generate(return_latent=True)
        pretty = {"casual": "Casual Scrollers", "conversation": "Conversation Starters",
                  "sharer": "Viral Sharers", "loyal": "Loyal Fans"}
        truth = np.array([pretty[t] for t in latent])
        print("\n" + "=" * 64 + "\nVALIDATION vs HIDDEN TRUTH (only possible with synthetic data!)\n" + "=" * 64)
        print(pd.crosstab(pd.Series(truth, name="hidden true type"),
                          pd.Series(seg_names, name="K-Means segment")).to_string())
        print(f"\nAdjusted Rand Index = {adjusted_rand_score(truth, seg_names):.3f}  (1.0 = perfect, 0 = random)")
        print(f"Segment named correctly for {(truth == seg_names).mean():.1%} of posts")
    except Exception as e:  # never let the optional check break training
        print(f"(Skipped hidden-truth validation: {e})")

    # ---- Figures + artefacts ----
    centroid_xy = {name: tuple(pca.transform(km.cluster_centers_[cid: cid + 1])[0]) for cid, name in id_to_name.items()}
    plot_audience_map(coords, seg_names, centroid_xy, var)
    plot_profiles(profile)

    joblib.dump({"scaler": scaler, "kmeans": km, "pca": pca, "id_to_name": id_to_name}, C.SEGMENTER_PATH)

    pd.DataFrame({
        "post_id": df.post_id, "pc1": coords[:, 0].round(4), "pc2": coords[:, 1].round(4),
        "segment": seg_names, "category": df.category, "engagement_rate": df.engagement_rate,
        "performance_label": df.performance_label, "views": df.views,
    }).to_csv(C.AUDIENCE_MAP_PATH, index=False)

    with open(C.SEGMENT_PROFILES_PATH, "w") as f:
        json.dump({
            "profiles": profile.round(3).to_dict(orient="index"),
            "pca_explained_variance": [float(v) for v in var],
            "silhouette": float(silhouette_score(X, km.labels_)),
            "centroid_pca_xy": {k: [float(v[0]), float(v[1])] for k, v in centroid_xy.items()},
            "segment_info": SEGMENT_INFO,
        }, f, indent=2)
    print(f"\nSaved segmenter -> {C.SEGMENTER_PATH}\nSaved audience map data -> {C.AUDIENCE_MAP_PATH}")

    demo_segment_prediction()


# ======================================================================
# Used by the Streamlit app: place ONE new post on the audience map
# ======================================================================
def predict_segment(likes, comments, shares, avg_watch_pct, bundle=None) -> dict:
    bundle = bundle or joblib.load(C.SEGMENTER_PATH)
    X = bundle["scaler"].transform(behaviour_features(likes, comments, shares, avg_watch_pct))
    cid = int(bundle["kmeans"].predict(X)[0])
    xy = bundle["pca"].transform(X)[0]
    return {"segment": bundle["id_to_name"][cid], "cluster_id": cid,
            "pc1": float(xy[0]), "pc2": float(xy[1]),
            "distance_to_centre": float(np.linalg.norm(X[0] - bundle["kmeans"].cluster_centers_[cid]))}


def demo_segment_prediction():
    print("\n" + "=" * 64 + "\nDEMO: assign new posts to a segment\n" + "=" * 64)
    bundle = joblib.load(C.SEGMENTER_PATH)
    examples = {
        "Mostly likes, quick watch":   dict(likes=900, comments=40, shares=40, avg_watch_pct=30),
        "Lots of comments":            dict(likes=600, comments=250, shares=90, avg_watch_pct=55),
        "Lots of shares":              dict(likes=500, comments=60, shares=300, avg_watch_pct=60),
        "Long watch, steady likes":    dict(likes=800, comments=120, shares=60, avg_watch_pct=85),
    }
    for label, kw in examples.items():
        r = predict_segment(bundle=bundle, **kw)
        print(f"{label:28s} -> {r['segment']}")


if __name__ == "__main__":
    main()