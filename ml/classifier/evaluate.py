"""Evaluate the NOMA-vs-OFDMA Random Forest classifier and save result figures.

Uses the same split and hyper-parameters as train_model.py (80:20, random_state=42,
150 trees) and writes the confusion matrix, feature importances and a text report
to results/.

Usage (from the repository root):
    python ml/classifier/evaluate.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

data = pd.read_csv(ROOT / "data" / "dataset.csv", header=None)
features = ["SNR (dB)", "g1 (weak user gain)", "g2 (strong user gain)", "alpha_1"]
X, y = data.iloc[:, 0:4], data.iloc[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=150, random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["OFDMA (0)", "NOMA (1)"])

with open(RESULTS_DIR / "classifier_report.txt", "w") as f:
    f.write(f"Accuracy: {acc:.3f}\n\nConfusion Matrix:\n{cm}\n\nReport:\n{report}")
print(f"Accuracy: {acc:.3f}\n{cm}\n{report}")

# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay(cm, display_labels=["OFDMA (0)", "NOMA (1)"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix (accuracy = {acc:.3f})")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
plt.close(fig)

# Feature importance
order = np.argsort(model.feature_importances_)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.barh(np.array(features)[order], model.feature_importances_[order], color="#2a6fbb")
ax.set_xlabel("Importance")
ax.set_title("Random Forest Feature Importance")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
plt.close(fig)
