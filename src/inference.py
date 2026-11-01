"""
inference.py
------------
Everything the dashboard needs that is NOT drawing: loading saved models, predicting one post,
explaining the prediction, and suggesting improvements. Kept separate from Streamlit so it can be
tested (and explained in a viva) on its own.

Main entry point:   analyze(inputs, artifacts)  ->  dict of results
"""
import json

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src.segmentation import SEGMENT_INFO, behaviour_features, predict_segment

FEATURE_LABELS = {
    "prev_avg_engagement": "Previous engagement",
    "trending_audio": "Trending audio",
    "duration_sec": "Video duration",
    "category": "Content category",
    "post_hour": "Posting hour",
    "hashtag_count": "Hashtag count",
    "caption_length": "Caption length",
    "post_day": "Posting day",
    "followers": "Follower count",
}

N_REFERENCE_ROWS = 200      # posts used as the "typical post" when explaining a prediction
MIN_GAIN = 0.20             # ignore suggestions predicted to help by less than 0.2 points


# ======================================================================
# Loading
# ======================================================================
def load_artifacts() -> dict:
    needed = [C.CLASSIFIER_PATH, C.REGRESSOR_PATH, C.SEGMENTER_PATH, C.METRICS_PATH, C.IMPORTANCE_PATH,
              C.AUDIENCE_MAP_PATH, C.SEGMENT_PROFILES_PATH, C.THRESHOLDS_PATH, C.DATASET_PATH]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing files: " + ", ".join(missing) + ".\nRun these from the project root first:\n"
            "  python -m src.generate_dataset\n  python -m src.train_models\n  python -m src.segmentation")

    df = pd.read_csv(C.DATASET_PATH)
    with open(C.METRICS_PATH) as f:
        metrics = json.load(f)
    with open(C.SEGMENT_PROFILES_PATH) as f:
        profiles = json.load(f)
    with open(C.THRESHOLDS_PATH) as f:
        thresholds = json.load(f)

    X = df[C.PRE_PUBLISH_FEATURES]
    return {
        "clf": joblib.load(C.CLASSIFIER_PATH),
        "reg": joblib.load(C.REGRESSOR_PATH),
        "segmenter": joblib.load(C.SEGMENTER_PATH),
        "metrics": metrics,
        "profiles": profiles,
        "thresholds": thresholds,
        "importance": pd.read_csv(C.IMPORTANCE_PATH, index_col="feature"),
        "audience_map": pd.read_csv(C.AUDIENCE_MAP_PATH),
        "er_sorted": np.sort(df["engagement_rate"].values),
        "reference": X.sample(N_REFERENCE_ROWS, random_state=C.RANDOM_SEED).reset_index(drop=True),
        "ranges": {c: (float(X[c].min()), float(X[c].max())) for c in C.NUMERIC_FEATURES},
        "n_posts": len(df),
    }


# ======================================================================
# Helpers
# ======================================================================
def to_frame(inputs: dict) -> pd.DataFrame:
    row = {f: inputs[f] for f in C.PRE_PUBLISH_FEATURES}
    row["trending_audio"] = int(bool(row["trending_audio"]))
    return pd.DataFrame([row])[C.PRE_PUBLISH_FEATURES]


def has_results(inputs: dict) -> bool:
    """True if the user typed in real post-publish numbers (views > 0 and some interactions)."""
    total = inputs.get("likes", 0) + inputs.get("comments", 0) + inputs.get("shares", 0)
    return inputs.get("views", 0) > 0 and total > 0


def _fmt(v) -> str:
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:g}"


def range_warnings(inputs: dict, art: dict) -> list:
    out = []
    for f, (lo, hi) in art["ranges"].items():
        v = inputs[f]
        if v < lo or v > hi:
            out.append(f"{FEATURE_LABELS[f]} = {_fmt(v)} is outside the range this model was built on "
                       f"({_fmt(lo)} to {_fmt(hi)}), so the estimate is less reliable.")
    return out


# ======================================================================
# Explanation: "what moved THIS prediction?"
# ======================================================================
def local_effects(x: pd.DataFrame, reg, reference: pd.DataFrame) -> pd.DataFrame:
    """
    For each feature: (prediction for this post) - (average prediction if ONLY that feature were
    replaced by values from typical posts). Positive = the feature pushed engagement UP.
    Simple, model-agnostic 'what-if' explanation (a rough cousin of SHAP; ignores feature interactions).
    """
    base = float(reg.predict(x)[0])
    n = len(reference)
    frames = []
    for f in C.PRE_PUBLISH_FEATURES:
        tmp = x.loc[x.index.repeat(n)].reset_index(drop=True)
        tmp[f] = reference[f].values
        frames.append(tmp)
    preds = reg.predict(pd.concat(frames, ignore_index=True)).reshape(len(C.PRE_PUBLISH_FEATURES), n)
    effects = base - preds.mean(axis=1)
    out = pd.DataFrame({"feature": C.PRE_PUBLISH_FEATURES, "effect": effects})
    out["label"] = out["feature"].map(FEATURE_LABELS)
    return out.sort_values("effect", key=np.abs, ascending=False).reset_index(drop=True)


# ======================================================================
# Insights: single-change what-if search
# ======================================================================
def _grid(feature, current):
    grids = {
        "duration_sec": list(range(5, 91, 5)),
        "caption_length": list(range(0, 301, 20)),
        "hashtag_count": list(range(0, 16)),
        "post_hour": list(range(24)),
        "post_day": C.DAYS,
        "trending_audio": [0, 1],
    }
    return [v for v in grids[feature] if v != current]


def _describe(feature, new, cur):
    if feature == "duration_sec":
        return f"{'Shorten' if new < cur else 'Extend'} the video to about {new} s (now {cur} s)"
    if feature == "caption_length":
        return f"Aim for a caption of about {new} characters (now {cur})"
    if feature == "hashtag_count":
        return f"Use about {new} hashtags (now {cur})"
    if feature == "post_hour":
        return f"Post at {new:02d}:00 instead of {cur:02d}:00"
    if feature == "post_day":
        return f"Post on {new} instead of {cur}"
    return "Use a trending audio track" if new == 1 else "Drop the trending audio (model sees no gain from it here)"


def recommendations(x: pd.DataFrame, reg, top_n=4) -> list:
    base = float(reg.predict(x)[0])
    candidates = []                       # (feature, value)
    for f in ["duration_sec", "caption_length", "hashtag_count", "post_hour", "post_day", "trending_audio"]:
        cur = x.iloc[0][f]
        for v in _grid(f, cur):
            candidates.append((f, v))
    frames = []
    for f, v in candidates:
        r = x.copy(); r[f] = v; frames.append(r)
    preds = reg.predict(pd.concat(frames, ignore_index=True))

    best = {}
    for (f, v), p in zip(candidates, preds):
        if f not in best or p > best[f][1]:
            best[f] = (v, float(p))
    recs = []
    for f, (v, p) in best.items():
        gain = p - base
        if gain >= MIN_GAIN:
            recs.append({"feature": f, "new_value": v.item() if hasattr(v, "item") else v,
                         "gain": gain, "text": _describe(f, v, x.iloc[0][f]),
                         "new_prediction": p})
    return sorted(recs, key=lambda r: -r["gain"])[:top_n]


def time_heatmap(x: pd.DataFrame, reg) -> pd.DataFrame:
    """Predicted engagement for every (day, hour) with all other inputs unchanged. Rows=days, cols=hours."""
    rows = []
    for d in C.DAYS:
        for h in range(24):
            r = x.copy(); r["post_day"] = d; r["post_hour"] = h; rows.append(r)
    preds = reg.predict(pd.concat(rows, ignore_index=True)).reshape(7, 24)
    return pd.DataFrame(preds, index=C.DAYS, columns=range(24))


# ======================================================================
# Words
# ======================================================================
HEADLINES = {"High": "Strong performer.", "Medium": "Middle of the pack.", "Low": "Likely to underperform."}


def narrative(label, er, pct, effects, mae) -> str:
    text = (f"The model expects about {er:.1f}% engagement (typically within {mae:.1f} points), "
            f"ahead of {pct:.0f}% of reference posts.")
    def phrase(r, positive):
        if r.feature == "trending_audio" and not positive:
            return "no trending audio"
        return r.label.lower()
    ups = [phrase(r, True) for r in effects.itertuples() if r.effect >= 0.2][:2]
    downs = [phrase(r, False) for r in effects.itertuples() if r.effect <= -0.2][:2]
    if ups:
        text += f" Helped most by {' and '.join(ups)}."
    if downs:
        text += f" Held back by {' and '.join(downs)}."
    return text


# ======================================================================
# Main
# ======================================================================
def analyze(inputs: dict, art: dict) -> dict:
    x = to_frame(inputs)
    clf, reg = art["clf"], art["reg"]

    proba = clf.predict_proba(x)[0]
    probs = {c: float(p) for c, p in zip(clf.classes_, proba)}
    label = max(probs, key=probs.get)
    er = float(reg.predict(x)[0])
    mae = float(art["metrics"]["regression"]["Random Forest"]["mae"])
    pct = float(np.searchsorted(art["er_sorted"], er) / len(art["er_sorted"]) * 100)

    effects = local_effects(x, reg, art["reference"])

    result = {
        "inputs": dict(inputs),
        "label": label,
        "probs": {k: probs.get(k, 0.0) for k in C.LABELS},
        "confidence": probs[label],
        "er_pred": er, "mae": mae, "percentile": pct,
        "headline": HEADLINES[label],
        "sub": narrative(label, er, pct, effects, mae),
        "effects": effects,
        "recs": recommendations(x, reg),
        "heat": time_heatmap(x, reg),
        "warnings": range_warnings(inputs, art),
        "segment": None, "er_actual": None,
    }
    if has_results(inputs):
        seg = predict_segment(inputs["likes"], inputs["comments"], inputs["shares"], inputs["watch"],
                              bundle=art["segmenter"])
        result["segment"] = seg
        result["er_actual"] = (inputs["likes"] + inputs["comments"] + inputs["shares"]) / inputs["views"] * 100
        if result["er_actual"] > 100:
            result["warnings"].append("Likes + comments + shares exceed views, so the engagement rate is above 100%. "
                                      "Check the numbers you typed.")
    return result


# ======================================================================
# Report content (used by src/report.py to write the downloadable PDF)
# ======================================================================
ACTION_WHY = {
    "duration_sec": "The model rewards tighter videos: engagement is highest for clips of roughly 15-30 seconds. "
                    "Cut slow intros and repeated beats.",
    "caption_length": "The model favours medium-length captions (roughly 40-150 characters). Lead with the hook, "
                      "then add one clear call to action.",
    "hashtag_count": "Results are best around 3-9 hashtags; none, or a wall of tags, does worse. "
                     "Choose specific tags over generic ones.",
    "post_hour": "Engagement is higher in the lunchtime and evening windows and lowest late at night. "
                 "Post when your audience is scrolling.",
    "post_day": "Some days score higher for posts like this one. The best-time map in this report shows the pattern.",
    "trending_audio": "Posts that use a trending sound get a clear lift in the model. "
                      "Pick a sound that genuinely fits the content.",
}

SEGMENT_PLAYBOOK = {
    "Casual Scrollers": [
        "Show the payoff in the first two seconds. Casual viewers decide before the third.",
        "Trim to the essentials and aim for the 15-30 second zone.",
        "Ask for one easy action (double-tap or save) instead of a long comment.",
        "Add on-screen text so the video still works with the sound off.",
    ],
    "Conversation Starters": [
        "End the caption with a specific question people can answer in one line.",
        "Reply to the first comments quickly. Early replies keep threads going.",
        "Try polls, 'this or that' choices or friendly opinion prompts.",
        "Pin a strong comment to steer the discussion.",
    ],
    "Viral Sharers": [
        "Make it relatable or useful enough to send to a friend (tips, lists, 'tag someone who...').",
        "Add a 'send this to someone who...' line to the caption or final frame.",
        "Use a trending sound; the model sees a clear lift from it.",
        "Keep it understandable without context so it survives being forwarded.",
    ],
    "Loyal Fans": [
        "Turn it into a series (Part 2, Part 3) so fans have a reason to come back.",
        "Post at a steady rhythm. Consistency is what this audience rewards.",
        "Give them something extra: behind-the-scenes clips, longer cuts, early access.",
        "Acknowledge regulars in the comments to strengthen the habit.",
    ],
}


def describe_value(feature: str, inputs: dict) -> str:
    v = inputs[feature]
    formatters = {
        "duration_sec": lambda x: f"{x} s",
        "caption_length": lambda x: f"{x} characters",
        "hashtag_count": lambda x: f"{x}",
        "post_hour": lambda x: f"{int(x):02d}:00",
        "post_day": str,
        "category": str,
        "trending_audio": lambda x: "used" if x else "not used",
        "followers": lambda x: f"{x:,.0f}",
        "prev_avg_engagement": lambda x: f"{x:.1f}%",
    }
    return formatters[feature](v)


def report_content(r: dict, art: dict) -> dict:
    """Everything the written report says, as plain data (so it can be tested without drawing anything)."""
    inp = r["inputs"]
    eff = r["effects"]
    rec_features = {x["feature"] for x in r["recs"]}

    lifts = [f"{e.label} ({describe_value(e.feature, inp)}): about {e.effect:+.1f} points vs a typical post"
             for e in eff.itertuples() if e.effect >= 0.2]
    drags = [f"{e.label} ({describe_value(e.feature, inp)}): about {e.effect:+.1f} points vs a typical post"
             for e in eff.itertuples() if e.effect <= -0.2]
    actions = [{"title": x["text"], "gain": x["gain"], "why": ACTION_WHY[x["feature"]]} for x in r["recs"]]
    keep = [f"{e.label}: {describe_value(e.feature, inp)}" for e in eff.itertuples()
            if e.effect >= 0.2 and e.feature not in rec_features]

    heat = r["heat"]
    best_day, best_hour = heat.stack().idxmax()
    best_val = float(heat.loc[best_day, best_hour])
    cur_val = float(heat.loc[inp["post_day"], inp["post_hour"]])
    best_time = {"day": best_day, "hour": int(best_hour), "value": best_val,
                 "current_day": inp["post_day"], "current_hour": int(inp["post_hour"]), "current_value": cur_val}

    audience = {"placed": False, "overview": [
        {"segment": s, "description": SEGMENT_INFO[s]["description"], "lever": SEGMENT_PLAYBOOK[s][0]}
        for s in SEGMENT_INFO]}
    seg = r["segment"]
    if seg:
        name = seg["segment"]
        prof = art["profiles"]["profiles"][name]
        f = behaviour_features(inp["likes"], inp["comments"], inp["shares"], inp["watch"]).iloc[0]
        cent = art["profiles"]["centroid_pca_xy"]
        others = sorted((s for s in cent if s != name),
                        key=lambda s: (cent[s][0] - seg["pc1"]) ** 2 + (cent[s][1] - seg["pc2"]) ** 2)
        audience = {
            "placed": True, "segment": name, "description": SEGMENT_INFO[name]["description"],
            "table": [
                ("Comments, % of interactions", float(f["comment_share"]), float(prof["comment_share"])),
                ("Shares, % of interactions", float(f["share_share"]), float(prof["share_share"])),
                ("Average watch time, %", float(f["avg_watch_pct"]), float(prof["avg_watch_pct"])),
            ],
            "tactics": SEGMENT_PLAYBOOK[name],
            "shift": [{"segment": s, "description": SEGMENT_INFO[s]["description"],
                       "tactic": SEGMENT_PLAYBOOK[s][0]} for s in others],
            "overview": audience["overview"],
        }

    return {"lifts": lifts, "drags": drags, "actions": actions, "keep": keep,
            "best_time": best_time, "audience": audience}