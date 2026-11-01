"""
train_models.py
---------------
Phase 2: trains, evaluates and saves
  1. a Random Forest CLASSIFIER  -> Low / Medium / High performance
  2. a Random Forest REGRESSOR   -> engagement rate (%)
and explains them with FEATURE IMPORTANCE.

Each Random Forest is compared with simple baselines, so we can prove the ML actually learned
something (a model is only "good" relative to a baseline!).

How to run (from the project root):
    python -m src.train_models
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, classification_report)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src import config as C
from src.preprocessing import build_preprocessor, get_train_test, load_data

SEED = C.RANDOM_SEED

# Hyper-parameter search space (kept small on purpose).
#  min_samples_leaf  : every leaf must contain at least N posts -> stops trees memorising noise
#  max_features      : how many features each split may look at -> makes trees more different
# NOTE: a first version used tiny leaves (1-5); it scored the same on test data but memorised the
# training set (train accuracy 100%). Larger leaves gave the same test score with far less overfitting.
RF_GRID = {"model__min_samples_leaf": [5, 10, 20], "model__max_features": ["sqrt", 0.5]}


# ======================================================================
# Helpers
# ======================================================================
def clf_metrics(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro")}


def reg_metrics(y_true, y_pred):
    return {"mae": mean_absolute_error(y_true, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": r2_score(y_true, y_pred)}


def as_pipeline(model, scale=False):
    return Pipeline([("prep", build_preprocessor(scale=scale)), ("model", model)])


def collapse_impurity_importance(pipe) -> pd.Series:
    """Random Forest gives one importance per ONE-HOT column (e.g. category_Comedy).
    We add them back together so we get one number per ORIGINAL feature (e.g. category)."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    imps = pipe.named_steps["model"].feature_importances_
    totals = {}
    for name, imp in zip(names, imps):
        if name.startswith("cat__category_"):
            key = "category"
        elif name.startswith("cat__post_day_"):
            key = "post_day"
        else:
            key = name.replace("num__", "")
        totals[key] = totals.get(key, 0) + imp
    return pd.Series(totals)


def section(title):
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)


# ======================================================================
# 1. Classification
# ======================================================================
def train_classifier(X_tr, X_te, y_tr, y_te):
    section("CLASSIFICATION: predict Low / Medium / High")

    baselines = {
        "Baseline: always predict majority": as_pipeline(DummyClassifier(strategy="most_frequent")),
        "Logistic Regression": as_pipeline(LogisticRegression(max_iter=2000), scale=True),
    }
    results = {}
    for name, pipe in baselines.items():
        pipe.fit(X_tr, y_tr)
        results[name] = clf_metrics(y_te, pipe.predict(X_te))

    # Random Forest with a small grid search (3-fold cross-validation on the TRAIN set only)
    rf = as_pipeline(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))
    grid = GridSearchCV(
        rf,
        param_grid=RF_GRID,
        cv=3, scoring="f1_macro", n_jobs=-1,
    )
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    print(f"Best hyper-parameters: {grid.best_params_}  (CV macro-F1 = {grid.best_score_:.3f})")

    pred = best.predict(X_te)
    results["Random Forest"] = clf_metrics(y_te, pred)
    train_acc = accuracy_score(y_tr, best.predict(X_tr))

    print("\nModel comparison on the TEST set:")
    print(pd.DataFrame(results).T.round(3).to_string())
    print(f"\nOverfitting check -> train accuracy {train_acc:.3f} vs test accuracy "
          f"{results['Random Forest']['accuracy']:.3f}")
    print("\nPer-class report (Random Forest):")
    print(classification_report(y_te, pred, labels=C.LABELS, digits=3))

    cm = confusion_matrix(y_te, pred, labels=C.LABELS)
    plot_confusion_matrix(cm)

    # Is the model's "confidence" trustworthy?
    proba = best.predict_proba(X_te)
    conf = proba.max(axis=1)
    correct = pred == y_te.values
    print(f"Mean confidence when correct: {conf[correct].mean():.2f} | when wrong: {conf[~correct].mean():.2f}")

    return best, results, cm, grid.best_params_


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Purples")
    ax.set_xticks(range(3)); ax.set_xticklabels(C.LABELS)
    ax.set_yticks(range(3)); ax.set_yticklabels(C.LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (test set)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "05_confusion_matrix.png", dpi=130); plt.close()


# ======================================================================
# 2. Regression
# ======================================================================
def train_regressor(X_tr, X_te, y_tr, y_te):
    section("REGRESSION: predict engagement rate (%)")

    baselines = {
        "Baseline: always predict the mean": as_pipeline(DummyRegressor(strategy="mean")),
        "Linear Regression": as_pipeline(LinearRegression(), scale=True),
    }
    results = {}
    for name, pipe in baselines.items():
        pipe.fit(X_tr, y_tr)
        results[name] = reg_metrics(y_te, pipe.predict(X_te))

    rf = as_pipeline(RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1))
    grid = GridSearchCV(
        rf,
        param_grid=RF_GRID,
        cv=3, scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    print(f"Best hyper-parameters: {grid.best_params_}  (CV MAE = {-grid.best_score_:.3f})")

    pred = best.predict(X_te)
    results["Random Forest"] = reg_metrics(y_te, pred)

    print("\nModel comparison on the TEST set (lower MAE/RMSE = better, higher R2 = better):")
    print(pd.DataFrame(results).T.round(3).to_string())
    print(f"\nOverfitting check -> train R2 {r2_score(y_tr, best.predict(X_tr)):.3f} vs test R2 "
          f"{results['Random Forest']['r2']:.3f}")

    plot_regression(y_te, pred)
    return best, results, grid.best_params_


def plot_regression(y_true, y_pred):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(y_true, y_pred, s=8, alpha=0.5, color="#8b5cf6")
    lo, hi = float(y_true.min()), float(y_true.max())
    ax[0].plot([lo, hi], [lo, hi], "k--", label="perfect prediction")
    ax[0].set_xlabel("Actual engagement (%)"); ax[0].set_ylabel("Predicted engagement (%)")
    ax[0].set_title("Predicted vs actual (test set)"); ax[0].legend()
    residuals = y_true - y_pred
    ax[1].hist(residuals, bins=40, color="#ec4899", alpha=0.85)
    ax[1].axvline(0, color="black", linestyle="--")
    ax[1].set_xlabel("Error = actual - predicted (%)"); ax[1].set_ylabel("Posts")
    ax[1].set_title("Residuals (should be centred on 0)")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "06_regression_performance.png", dpi=130); plt.close()


# ======================================================================
# 3. Feature importance
# ======================================================================
def feature_importance(clf, reg, X_te, yc_te, yr_te):
    section("FEATURE IMPORTANCE: what drives the predictions?")
    rows = {}
    # (a) built-in impurity importance (fast, but can favour numeric features)
    rows["clf_impurity"] = collapse_impurity_importance(clf)
    rows["reg_impurity"] = collapse_impurity_importance(reg)
    # (b) permutation importance: shuffle ONE column of the test set and measure how much the score
    #     drops. Big drop = the model relied on that feature. Works on the raw columns directly.
    pi_c = permutation_importance(clf, X_te, yc_te, n_repeats=10, random_state=SEED,
                                  scoring="f1_macro", n_jobs=-1)
    pi_r = permutation_importance(reg, X_te, yr_te, n_repeats=10, random_state=SEED,
                                  scoring="r2", n_jobs=-1)
    rows["clf_permutation"] = pd.Series(pi_c.importances_mean, index=X_te.columns)
    rows["reg_permutation"] = pd.Series(pi_r.importances_mean, index=X_te.columns)
    rows["reg_permutation_std"] = pd.Series(pi_r.importances_std, index=X_te.columns)

    imp = pd.DataFrame(rows).sort_values("reg_permutation", ascending=False)
    imp.index.name = "feature"
    imp.to_csv(C.IMPORTANCE_PATH)
    print(imp.round(3).to_string())

    # Plot: permutation importance for both models
    order = imp.sort_values("reg_permutation").index
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    ax[0].barh(order, imp.loc[order, "clf_permutation"], color="#8b5cf6")
    ax[0].set_title("Classifier: drop in macro-F1 when shuffled")
    ax[1].barh(order, imp.loc[order, "reg_permutation"], xerr=imp.loc[order, "reg_permutation_std"],
               color="#ec4899")
    ax[1].set_title("Regressor: drop in R2 when shuffled")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "07_feature_importance.png", dpi=130); plt.close()
    return imp


# ======================================================================
# 4. Save + demo
# ======================================================================
def demo_prediction():
    """Loads the saved models from disk and predicts one example post - exactly what the
    Streamlit app will do in Phase 4."""
    section("DEMO: load saved models and predict one post")
    clf = joblib.load(C.CLASSIFIER_PATH)
    reg = joblib.load(C.REGRESSOR_PATH)
    post = pd.DataFrame([{
        "duration_sec": 20, "caption_length": 80, "hashtag_count": 6, "post_hour": 20,
        "post_day": "Sat", "category": "Comedy", "trending_audio": 1,
        "followers": 15000, "prev_avg_engagement": 7.0,
    }])[C.PRE_PUBLISH_FEATURES]
    proba = clf.predict_proba(post)[0]
    print("Input:", post.iloc[0].to_dict())
    print(f"Predicted class : {clf.predict(post)[0]}")
    print("Probabilities   :", {c: round(float(p), 3) for c, p in zip(clf.classes_, proba)})
    print(f"Confidence      : {proba.max():.0%}")
    print(f"Predicted engagement rate: {reg.predict(post)[0]:.2f}%")


def main():
    C.MODELS_DIR.mkdir(exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = get_train_test(df)
    print(f"Train: {len(X_tr):,} posts | Test: {len(X_te):,} posts | "
          f"Features used: {list(X_tr.columns)}")

    clf, clf_res, cm, clf_params = train_classifier(X_tr, X_te, yc_tr, yc_te)
    reg, reg_res, reg_params = train_regressor(X_tr, X_te, yr_tr, yr_te)
    imp = feature_importance(clf, reg, X_te, yc_te, yr_te)

    joblib.dump(clf, C.CLASSIFIER_PATH)
    joblib.dump(reg, C.REGRESSOR_PATH)
    with open(C.METRICS_PATH, "w") as f:
        json.dump({
            "classification": clf_res, "regression": reg_res,
            "confusion_matrix": {"labels": C.LABELS, "matrix": cm.tolist()},
            "best_params": {"classifier": clf_params, "regressor": reg_params},
            "n_train": len(X_tr), "n_test": len(X_te),
        }, f, indent=2, default=float)
    print(f"\nSaved models -> {C.MODELS_DIR}\nSaved figures -> {C.FIG_DIR}")

    demo_prediction()


if __name__ == "__main__":
    main()