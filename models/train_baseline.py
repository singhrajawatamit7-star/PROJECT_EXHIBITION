"""
Baseline Machine Learning Classifier Training
Trains a Random Forest classifier on engineered facial and head-pose telemetry features.
Exports model to models/trained/drowsiness_random_forest.joblib
"""

import sys
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "features.csv"
MODEL_OUTPUT_PATH = ROOT_DIR / "models" / "trained" / "drowsiness_random_forest.joblib"


def train_baseline_model():
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, roc_auc_score
        import joblib
    except ImportError as e:
        print(f"[ML Training] Missing required dependencies ({e}). Install scikit-learn, pandas, joblib.")
        return False

    if not DATA_PATH.exists():
        print(f"[ML Training] Dataset not found at {DATA_PATH}. Generating synthetic dataset first...")
        sys.path.insert(0, str(ROOT_DIR / "app"))
        from simulate import generate_feature_dataset
        generate_feature_dataset(output_path=DATA_PATH, num_samples=3000)

    print(f"[ML Training] Loading features from {DATA_PATH}...")
    data = pd.read_csv(DATA_PATH)

    feature_columns = [
        "ear",
        "perclos_30s",
        "mar",
        "blink_duration",
        "head_pitch",
        "head_yaw"
    ]

    X = data[feature_columns]
    y = data["label"]  # 0 = alert, 1 = drowsy

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"[ML Training] Training Random Forest with {len(X_train)} train samples and {len(X_test)} test samples...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, predictions, target_names=["Alert", "Drowsy"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")

    # Feature Importances
    importances = model.feature_importances_
    print("\n--- FEATURE IMPORTANCES ---")
    for feat, imp in sorted(zip(feature_columns, importances), key=lambda x: x[1], reverse=True):
        print(f"  {feat:20s}: {imp * 100:.2f}%")

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    joblib.dump(model, ROOT_DIR / "models" / "trained" / "drowsiness_model.joblib")
    print(f"\n[ML Training] Saved trained baseline model to: {MODEL_OUTPUT_PATH} and drowsiness_model.joblib")
    return True


if __name__ == "__main__":
    train_baseline_model()
