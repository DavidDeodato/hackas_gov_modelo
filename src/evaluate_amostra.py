#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Avaliação/diagnóstico na amostra e-SIC (mascarada).

IMPORTANTE:
- A amostra `AMOSTRA_e-SIC - Amostra - SIC.csv` não possui rótulo, então não é
  possível calcular Precisão/Recall/F1 nela sem um y_true.
- Este script gera um relatório objetivo do comportamento do modelo na amostra:
  taxa de positivos, distribuição de probabilidades, exemplos mais prováveis,
  e exporta um CSV com as predições.

Uso (PowerShell):
  py .\\src\\evaluate_amostra.py
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_threshold(path: str) -> float:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return float(json.load(f).get("threshold", 0.5))
    except FileNotFoundError:
        return 0.5


def main() -> None:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Diagnóstico do modelo na amostra e-SIC (sem rótulo).")
    parser.add_argument(
        "--amostra",
        type=str,
        default=os.path.join(root, "AMOSTRA_e-SIC - Amostra - SIC.csv"),
        help="Caminho do CSV da amostra (padrão: AMOSTRA_e-SIC - Amostra - SIC.csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(root, "models", "modelo_tfidf.pkl"),
        help="Caminho do modelo .pkl",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        default=os.path.join(root, "models", "threshold.json"),
        help="Caminho do threshold.json",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=os.path.join(root, "docs", "amostra_predictions.csv"),
        help="CSV de saída com probabilidades e decisão",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default=os.path.join(root, "docs", "amostra_relatorio.md"),
        help="Relatório Markdown de saída",
    )
    parser.add_argument(
        "--out-fig",
        type=str,
        default=os.path.join(root, "docs", "amostra_proba_hist.png"),
        help="Figura: histograma das probabilidades",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="CSV com rótulos manuais (colunas: ID, contem_dados_pessoais [0/1])",
    )
    parser.add_argument(
        "--export-label-template",
        type=str,
        default=None,
        help="Gera um template de rotulagem (ID + contem_dados_pessoais vazio) e encerra",
    )
    parser.add_argument(
        "--out-metrics",
        type=str,
        default=os.path.join(root, "docs", "amostra_metricas.json"),
        help="Saída JSON com métricas (se --labels for fornecido)",
    )
    parser.add_argument(
        "--out-cm",
        type=str,
        default=os.path.join(root, "docs", "amostra_confusion_matrix.png"),
        help="Figura: matriz de confusão (se --labels for fornecido)",
    )
    parser.add_argument("--topk", type=int, default=15, help="Quantidade de exemplos para mostrar no relatório")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    df = pd.read_csv(args.amostra)
    if "Texto Mascarado" not in df.columns:
        raise ValueError("A amostra precisa ter a coluna 'Texto Mascarado'")

    # Exportar template de rotulagem e encerrar
    if args.export_label_template:
        tmpl = df[["ID"]].copy()
        tmpl["contem_dados_pessoais"] = ""
        os.makedirs(os.path.dirname(os.path.abspath(args.export_label_template)), exist_ok=True)
        tmpl.to_csv(args.export_label_template, index=False)
        print("OK:", args.export_label_template)
        return

    texts = df["Texto Mascarado"].astype(str).fillna("").values

    model = joblib.load(args.model)
    thr = _load_threshold(args.threshold)

    proba = model.predict_proba(texts)[:, 1]
    pred = (proba >= thr).astype(int)

    out = df.copy()
    out["probabilidade_com_dados"] = proba
    out["predito_contem_dados_pessoais"] = pred
    out.to_csv(args.out_csv, index=False)

    pos_rate = float(pred.mean()) if len(pred) else 0.0

    # Figura: histograma
    plt.figure(figsize=(6, 4))
    plt.hist(proba, bins=30, alpha=0.8)
    plt.axvline(thr, linestyle="--", linewidth=1, color="black", label=f"threshold={thr:.2f}")
    plt.title("Amostra e-SIC: distribuição de P(y=1)")
    plt.xlabel("P(y=1)")
    plt.ylabel("Contagem")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=200)
    plt.close()

    # Top exemplos
    idx_sorted = np.argsort(proba)
    top_pos = idx_sorted[::-1][: args.topk]
    top_neg = idx_sorted[: args.topk]

    def _fmt_examples(indices):
        lines = []
        for i in indices:
            txt = str(out.iloc[i]["Texto Mascarado"]).replace("\n", " ")
            lines.append(f"- ID={int(out.iloc[i]['ID'])} | p={proba[i]:.4f} | {txt[:220]}")
        return "\n".join(lines)

    md = []
    md.append("### Diagnóstico na amostra e-SIC (sem rótulo)\n")
    md.append("#### Observação importante\n")
    md.append("A amostra não possui rótulo. Portanto, não é possível calcular Precisão/Recall/F1 nela.\n")
    md.append("O objetivo deste relatório é descrever o comportamento do modelo na amostra (distribuições e exemplos).\n")
    md.append("\n#### Resumo\n")
    md.append(f"- Total de registros: {len(out)}\n")
    md.append(f"- Threshold usado: {thr:.4f}\n")
    md.append(f"- Percentual predito como contendo dados pessoais: {pos_rate:.2%}\n")
    md.append(f"- CSV de predições: `{os.path.relpath(args.out_csv, root)}`\n")
    md.append(f"- Figura (histograma): `{os.path.relpath(args.out_fig, root)}`\n")
    md.append("\n#### Exemplos com maior probabilidade (top positivos)\n")
    md.append(_fmt_examples(top_pos))
    md.append("\n\n#### Exemplos com menor probabilidade (top negativos)\n")
    md.append(_fmt_examples(top_neg))
    md.append("\n")

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # Se houver rótulos manuais, calcular métricas
    if args.labels:
        labels_df = pd.read_csv(args.labels)
        if "ID" not in labels_df.columns or "contem_dados_pessoais" not in labels_df.columns:
            raise ValueError("Arquivo de rótulos deve ter colunas: ID, contem_dados_pessoais")

        merged = out.merge(labels_df[["ID", "contem_dados_pessoais"]], on="ID", how="inner")
        y_true = merged["contem_dados_pessoais"].astype(int).values
        y_pred = merged["predito_contem_dados_pessoais"].astype(int).values

        metrics = {
            "n_rotulados": int(len(merged)),
            "threshold": float(thr),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["confusion_matrix"] = {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}

        with open(args.out_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # figura matriz de confusão
        plt.figure(figsize=(5, 4))
        cm = np.array([[tn, fp], [fn, tp]])
        plt.imshow(cm, cmap="Blues")
        plt.title("Matriz de confusão (amostra e-SIC rotulada)")
        plt.xticks([0, 1], ["Sem dados", "Com dados"])
        plt.yticks([0, 1], ["Sem dados", "Com dados"])
        for (r, c), v in np.ndenumerate(cm):
            plt.text(c, r, str(v), ha="center", va="center")
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        plt.tight_layout()
        plt.savefig(args.out_cm, dpi=200)
        plt.close()

        print("OK:", args.out_metrics)
        print("OK:", args.out_cm)

    print("OK:", args.out_csv)
    print("OK:", args.out_md)
    print("OK:", args.out_fig)


if __name__ == "__main__":
    main()


