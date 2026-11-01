"""
check_setup.py
--------------
Tells you what is missing or wrong in your project BEFORE you launch the dashboard.

Run from the project root:
    python -m src.check_setup

It reads the source files as text (it does not import them), so it still works when a file is
incomplete or contains the wrong code, which is exactly when you need it.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# file -> names that must be defined at the top level of that file
REQUIRED_CODE = {
    "src/config.py": ["ROOT", "DATASET_PATH", "PRE_PUBLISH_FEATURES", "CLASSIFIER_PATH", "REGRESSOR_PATH",
                      "CLUSTER_FEATURES", "K_CLUSTERS", "SEGMENTER_PATH", "SEGMENT_PROFILES_PATH",
                      "AUDIENCE_MAP_PATH"],
    "src/generate_dataset.py": ["generate", "main"],
    "src/preprocessing.py": ["load_data", "get_train_test", "build_preprocessor"],
    "src/train_models.py": ["train_classifier", "train_regressor", "feature_importance", "main"],
    "src/segmentation.py": ["SEGMENT_INFO", "behaviour_features", "evaluate_k", "name_clusters",
                            "predict_segment", "main"],
    "src/inference.py": ["load_artifacts", "analyze", "local_effects", "recommendations", "time_heatmap",
                         "report_content", "SEGMENT_PLAYBOOK"],
    "src/report.py": ["Doc", "build_report_pdf", "draw_effects", "draw_heat", "draw_map"],
    "app/streamlit_app.py": ["get_artifacts", "build_map", "toggle_group", "panel", "store_result"],
}

# artefact -> command that creates it
ARTEFACTS = {
    "data/raw/synthetic_social_posts.csv": "python -m src.generate_dataset",
    "data/processed/label_thresholds.json": "python -m src.generate_dataset",
    "models/performance_classifier.joblib": "python -m src.train_models",
    "models/engagement_regressor.joblib": "python -m src.train_models",
    "models/metrics.json": "python -m src.train_models",
    "models/feature_importance.csv": "python -m src.train_models",
    "models/audience_segmenter.joblib": "python -m src.segmentation",
    "models/segment_profiles.json": "python -m src.segmentation",
    "data/processed/audience_map.csv": "python -m src.segmentation",
}


def defined_names(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def main():
    problems, commands = [], []
    print(f"Project folder: {ROOT}\n")

    print("1. Source files")
    for rel, needed in REQUIRED_CODE.items():
        path = ROOT / rel
        if not path.exists():
            print(f"  [MISSING] {rel}")
            problems.append(f"{rel} is missing - copy it from the latest zip")
            continue
        try:
            found = defined_names(path)
        except SyntaxError as e:
            print(f"  [BROKEN ] {rel}: syntax error on line {e.lineno} - file is incomplete or has wrong text in it")
            problems.append(f"{rel} has a syntax error - replace it with the file from the zip")
            continue
        absent = [n for n in needed if n not in found]
        size = path.stat().st_size
        if absent:
            print(f"  [WRONG  ] {rel} ({size:,} bytes) is missing: {', '.join(absent)}")
            problems.append(f"{rel} is incomplete or contains the wrong code - replace it with the file from the zip")
        else:
            print(f"  [OK     ] {rel} ({size:,} bytes)")

    print("\n2. Data and trained models")
    for rel, cmd in ARTEFACTS.items():
        if (ROOT / rel).exists():
            print(f"  [OK     ] {rel}")
        else:
            print(f"  [MISSING] {rel}   <- created by: {cmd}")
            if cmd not in commands:
                commands.append(cmd)

    for rel in ["requirements.txt", ".streamlit/config.toml"]:
        if not (ROOT / rel).exists():
            print(f"\n  [MISSING] {rel}")
            problems.append(f"{rel} is missing")

    print("\n" + "=" * 60)
    if problems:
        print("FIX THESE FIRST:")
        for p in problems:
            print("  -", p)
        print("Then run this checker again.")
        sys.exit(1)
    if commands:
        print("Source files look right. Now run, in this order:")
        for c in ["python -m src.generate_dataset", "python -m src.train_models", "python -m src.segmentation"]:
            if c in commands:
                print("  ", c)
        sys.exit(1)
    print("Everything is in place. Start the dashboard with:")
    print("   streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()