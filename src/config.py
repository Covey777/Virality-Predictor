"""
Central configuration for the whole project.
Every other file imports paths and constants from here, so nothing is hard-coded twice.
"""
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

DATASET_PATH = DATA_RAW / "synthetic_social_posts.csv"
THRESHOLDS_PATH = DATA_PROCESSED / "label_thresholds.json"

# ---------- Reproducibility ----------
RANDOM_SEED = 42
N_POSTS = 5000

# ---------- Domain constants ----------
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
LABELS = ["Low", "Medium", "High"]

# Category profile: (base engagement %, typical duration in sec, share of posts)
CATEGORY_PROFILE = {
    "Comedy":    (7.0, 22, 0.16),
    "Fitness":   (6.0, 35, 0.13),
    "Food":      (6.5, 40, 0.12),
    "Education": (5.0, 55, 0.12),
    "Travel":    (5.5, 30, 0.11),
    "Fashion":   (5.2, 20, 0.12),
    "Tech":      (4.2, 50, 0.08),
    "Music":     (7.5, 25, 0.10),
    "Gaming":    (4.8, 45, 0.06),
}
CATEGORIES = list(CATEGORY_PROFILE.keys())

# Hidden audience "behaviour types" used ONLY to generate realistic data.
# (like share, comment share, share share), mean watch-through %, probability
# K-Means in Phase 3 will try to rediscover this structure WITHOUT being told about it.
AUDIENCE_TYPES = {
    "casual":       ((0.90, 0.05, 0.05), 35, 0.35),
    "conversation": ((0.70, 0.22, 0.08), 55, 0.25),
    "sharer":       ((0.60, 0.08, 0.32), 60, 0.15),
    "loyal":        ((0.82, 0.12, 0.06), 78, 0.25),
}

# ---------- Feature groups (important for avoiding data leakage!) ----------
# Known BEFORE the post goes live -> legitimate inputs for prediction
PRE_PUBLISH_FEATURES = [
    "duration_sec", "caption_length", "hashtag_count", "post_hour", "post_day",
    "category", "trending_audio", "followers", "prev_avg_engagement",
]
# Only known AFTER the post is live -> used to build targets and audience segments
POST_PUBLISH_METRICS = ["views", "likes", "comments", "shares", "avg_watch_pct"]

# ---------- Column types (used by preprocessing) ----------
CATEGORICAL_FEATURES = ["category", "post_day"]
NUMERIC_FEATURES = [f for f in PRE_PUBLISH_FEATURES if f not in CATEGORICAL_FEATURES]

# ---------- Saved artefacts ----------
CLASSIFIER_PATH = MODELS_DIR / "performance_classifier.joblib"
REGRESSOR_PATH = MODELS_DIR / "engagement_regressor.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"
TEST_SIZE = 0.2

# ---------- Phase 3: audience segmentation ----------
# Behaviour features describe HOW an audience reacted (not how many people reacted):
#   comment_share = comments / (likes+comments+shares) * 100
#   share_share   = shares   / (likes+comments+shares) * 100
#   avg_watch_pct = average % of the video that viewers watched
CLUSTER_FEATURES = ["comment_share", "share_share", "avg_watch_pct"]
K_CLUSTERS = 4
K_RANGE = range(2, 9)
SEGMENTER_PATH = MODELS_DIR / "audience_segmenter.joblib"
SEGMENT_PROFILES_PATH = MODELS_DIR / "segment_profiles.json"
AUDIENCE_MAP_PATH = DATA_PROCESSED / "audience_map.csv"