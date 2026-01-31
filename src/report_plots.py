#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gera figuras de avaliação (confusion matrix e ROC) a partir do modelo exportado.

Uso (PowerShell):
  py .\\src\\report_plots.py
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    root = _project_root()
    dataset_path = os.path.join(root, "data", "dataset_pedidos_lai_aug.csv")
    model_path = os.path.join(root, "models", "modelo_tfidf.pkl")
    thr_path = os.path.join(root, "models", "threshold.json")

    out_dir = os.path.join(root, "docs")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    threshold = 0.5
    if os.path.exists(thr_path):
        with open(thr_path, "r", encoding="utf-8") as f:
            threshold = float(json.load(f).get("threshold", 0.5))

    df = pd.read_csv(dataset_path)
    X = df["pedido"].astype(str).fillna("").values
    y = df["contem_dados_pessoais"].astype(int).values

    seed = 42
    _, X_tmp, _, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    _, X_test, _, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
    )

    model = joblib.load(model_path)
    proba_test = model.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Sem dados", "Com dados"],
        yticklabels=["Sem dados", "Com dados"],
    )
    plt.title("Matriz de Confusão (TF‑IDF + LogReg)")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    cm_path = os.path.join(out_dir, "confusion_matrix_tfidf.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.title("Curva ROC (TF‑IDF + LogReg)")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    roc_path = os.path.join(out_dir, "roc_curve_tfidf.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    print("OK:", cm_path)
    print("OK:", roc_path)

    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, proba_test)
    ap = average_precision_score(y_test, proba_test)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.title("Curva Precision-Recall (TF‑IDF + LogReg)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    pr_path = os.path.join(out_dir, "pr_curve_tfidf.png")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=200)
    plt.close()

    # Histogram of probabilities by class
    plt.figure(figsize=(6, 4))
    sns.histplot(proba_test[y_test == 0], bins=30, stat="density", label="Sem dados", color="#1f77b4", alpha=0.5)
    sns.histplot(proba_test[y_test == 1], bins=30, stat="density", label="Com dados", color="#d62728", alpha=0.5)
    plt.axvline(threshold, linestyle="--", linewidth=1, color="black", label=f"threshold={threshold:.2f}")
    plt.title("Distribuição de probabilidade por classe (teste)")
    plt.xlabel("P(y=1)")
    plt.ylabel("Densidade")
    plt.legend()
    proba_hist_path = os.path.join(out_dir, "proba_hist_tfidf.png")
    plt.tight_layout()
    plt.savefig(proba_hist_path, dpi=200)
    plt.close()

    # Top coefficients (interpretabilidade)
    try:
        feats = model.named_steps["features"]
        clf = model.named_steps["clf"]
        word_vec = feats.transformer_list[0][1]
        char_vec = feats.transformer_list[1][1]
        names_word = [f"word:{t}" for t in word_vec.get_feature_names_out()]
        names_char = [f"char:{t}" for t in char_vec.get_feature_names_out()]
        feature_names = np.array(names_word + names_char)
        coefs = clf.coef_.ravel()

        top_k = 20
        idx_pos = np.argsort(coefs)[-top_k:][::-1]
        idx_neg = np.argsort(coefs)[:top_k]

        labels = np.concatenate([feature_names[idx_pos], feature_names[idx_neg]])
        values = np.concatenate([coefs[idx_pos], coefs[idx_neg]])

        plt.figure(figsize=(10, 7))
        sns.barplot(x=values, y=labels, orient="h")
        plt.axvline(0, color="black", linewidth=1)
        plt.title("Top termos por coeficiente (classe 1 vs classe 0)")
        plt.xlabel("Coeficiente (LogReg)")
        plt.ylabel("Termo")
        top_feat_path = os.path.join(out_dir, "top_features_tfidf.png")
        plt.tight_layout()
        plt.savefig(top_feat_path, dpi=200)
        plt.close()
    except Exception as e:
        top_feat_path = None
        print("WARN: não foi possível gerar top_features_tfidf.png:", e)

    print("OK:", pr_path)
    print("OK:", proba_hist_path)
    if top_feat_path:
        print("OK:", top_feat_path)


if __name__ == "__main__":
    main()


