"""Model training and SHAP explainability for Fraud Detection Studio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data_gen import FEATURES, build_training_data

# feature -> plain-English template used in the "why was this flagged" panel
EXPLANATIONS = {
    "txns_last_hour": "Unusual burst of activity — {v:.0f} transactions in the past hour",
    "mins_since_prev": "Only {v:.1f} minutes since the previous transaction",
    "dist_from_home_km": "Charged {v:,.0f} km away from the customer's home city",
    "amount_ratio": "Amount is {v:.1f}× this customer's typical spend",
    "amount": "Large amount: A${v:,.2f}",
    "log_amount": "Amount well outside this customer's normal range",
    "merchant_risk": "High-risk merchant category (risk weight {v:.2f})",
    "is_night": "Transaction made in the middle of the night",
    "is_online": "Card-not-present (online) transaction",
    "is_new_merchant": "First time this customer has used this merchant",
    "hour": "Unusual time of day for this customer",
}


def train_model(seed: int = 42) -> dict:
    """Train the classifier on simulated history; return model + explainer + metrics."""
    df = build_training_data(seed=seed)
    X, y = df[FEATURES], df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="aucpr",
        random_state=seed,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "fraud_rate": float(y.mean()),
    }

    explainer = shap.TreeExplainer(model)
    return {"model": model, "explainer": explainer, "metrics": metrics}


def score(model: XGBClassifier, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES])[:, 1]


def explain_row(explainer: shap.TreeExplainer, row: pd.Series) -> shap.Explanation:
    X = row[FEATURES].to_frame().T.astype(float)
    return explainer(X)[0]


def plain_english(explanation: shap.Explanation, row: pd.Series, top_k: int = 4) -> list[str]:
    """Top positive SHAP contributors rendered as human-readable reasons."""
    contrib = sorted(
        zip(FEATURES, explanation.values),
        key=lambda fv: fv[1],
        reverse=True,
    )
    reasons = []
    for feat, val in contrib[:top_k]:
        if val <= 0:
            continue
        template = EXPLANATIONS.get(feat, feat)
        reasons.append(template.format(v=float(row[feat])))
    return reasons
