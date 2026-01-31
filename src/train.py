#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Treino e geração de dataset para o hackathon (Categoria Acesso à Informação).

Foco: modelo rápido e robusto em CPU:
- TF‑IDF (char + word n-grams)
- Classificador linear (Logistic Regression)
- Threshold tuning para maximizar F1 (P1 do edital)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

# Permitir execução direta: `py .\src\train.py ...`
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from features_pii import scrub_pii  # noqa: E402


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _read_amostra_texts(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    col = None
    for c in df.columns:
        if c.strip().lower() in ("texto mascarado", "texto_mascarado", "texto"):
            col = c
            break
    if col is None:
        return []
    texts = df[col].astype(str).fillna("").tolist()
    return [t for t in texts if t.strip()]


def _random_digits(n: int) -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(n))


def _cpf_checksum(digits9: List[int]) -> Tuple[int, int]:
    # DV1
    s1 = sum(d * w for d, w in zip(digits9, range(10, 1, -1)))
    r1 = s1 % 11
    dv1 = 0 if r1 < 2 else 11 - r1
    # DV2
    digits10 = digits9 + [dv1]
    s2 = sum(d * w for d, w in zip(digits10, range(11, 1, -1)))
    r2 = s2 % 11
    dv2 = 0 if r2 < 2 else 11 - r2
    return dv1, dv2


def gerar_cpf_valido() -> str:
    digits9 = [random.randint(0, 9) for _ in range(9)]
    dv1, dv2 = _cpf_checksum(digits9)
    digits = digits9 + [dv1, dv2]
    cpf = "".join(map(str, digits))
    return f"{cpf[0:3]}.{cpf[3:6]}.{cpf[6:9]}-{cpf[9:11]}"


def gerar_cpf_invalido_11dig() -> str:
    # gera 11 dígitos que quase sempre NÃO passam no checksum
    # (e sem contexto de "CPF" nos hard negatives).
    return _random_digits(11)


def _pick(lst: List[str]) -> str:
    return random.choice(lst)


NOMES = [
    "João", "Maria", "Ana", "Pedro", "Paulo", "Carla", "Beatriz", "Rafael", "Juliana", "Gustavo",
    "Rodrigo", "Isabela", "Camila", "Bruno", "Larissa", "Thiago", "Mariana", "Felipe", "Luciana", "Eduardo",
]
SOBRENOMES = [
    "Silva", "Souza", "Oliveira", "Santos", "Pereira", "Costa", "Rodrigues", "Almeida", "Nunes", "Lima",
    "Carvalho", "Ferreira", "Gomes", "Ribeiro", "Martins", "Araújo", "Barbosa", "Cardoso", "Teixeira", "Vieira",
]
RUA = ["Rua das Flores", "Avenida Central", "Quadra L2 Sul", "Travessa Beira Rio", "Rua do Sol", "SQN 112"]
BAIRRO = ["Asa Norte", "Asa Sul", "Sudoeste", "Sobradinho", "Planaltina", "Santa Maria", "Brazlândia"]


def gerar_email(nome: str, sobrenome: str) -> str:
    doms = ["gmail.com", "hotmail.com", "outlook.com", "uol.com.br", "yahoo.com.br"]
    base = f"{nome}.{sobrenome}".lower()
    base = re.sub(r"[^a-z.]", "", base)
    return f"{base}@{_pick(doms)}"


def gerar_email_ofuscado(email: str) -> str:
    # variações vistas no mundo real
    variants = [
        email.replace("@", " [arroba] "),
        email.replace("@", " (at) "),
        email.replace("@", " at "),
    ]
    return _pick(variants)


def gerar_telefone_fmt() -> str:
    ddd = random.randint(11, 99)
    n1 = random.randint(90000, 99999)
    n2 = random.randint(1000, 9999)
    return f"({ddd}) {n1}-{n2}"


def gerar_telefone_espacos() -> str:
    ddd = random.randint(11, 99)
    n1 = random.randint(90000, 99999)
    n2 = random.randint(1000, 9999)
    return f"{ddd} {n1} {n2}"


def gerar_cep() -> str:
    return f"{random.randint(10000, 99999)}-{random.randint(100, 999)}"


def _positivos_sinteticos(n: int) -> List[str]:
    out: List[str] = []
    for _ in range(n):
        nome = _pick(NOMES)
        sobrenome = _pick(SOBRENOMES)
        nome_completo = f"{nome} {_pick(SOBRENOMES)} {sobrenome}"

        cpf = gerar_cpf_valido()
        email = gerar_email(nome, sobrenome)
        email_of = gerar_email_ofuscado(email)
        tel = gerar_telefone_fmt()
        tel2 = gerar_telefone_espacos()
        cep = gerar_cep()
        rua = _pick(RUA)
        bairro = _pick(BAIRRO)

        templates = [
            f"Meu nome é {nome_completo}. Solicito informações sobre meu processo.",
            f"Me chamo {nome_completo}. Solicito acesso aos autos do meu processo SEI.",
            f"Nome: {nome_completo}. Solicito acesso aos meus dados cadastrais.",
            f"Solicito a exclusão dos meus dados pessoais. Nome: {nome_completo}.",
            f"Solicito retificação cadastral. Meu CPF é {cpf}.",
            f"CPF: {cpf}. Solicito informações sobre meu cadastro.",
            f"CPF: {cpf.replace('.', ' ').replace('-', ' ')}. Solicito informações.",
            f"Contato: {tel}. Solicito retorno.",
            f"Telefone: {tel2}. Solicito retorno.",
            f"E-mail: {email}. Solicito acesso às minhas informações.",
            f"Meu e-mail é {email_of}. Solicito acesso às minhas informações.",
            f"Endereço: {rua}, {bairro}, Brasília-DF. CEP {cep}. Solicito atualização cadastral.",
            f"Meu endereço é {rua}, nº {random.randint(10, 9999)}. Solicito atualização.",
            f"Dados pessoais: Nome {nome_completo}, CPF {cpf}, Email {email}. Solicito anonimização.",
            f"Sou {nome_completo}, titular do CPF {cpf} e título de eleitor nº {random.randint(100000000000, 999999999999)}. Solicito informações eleitorais.",
            f"Meu nome é {nome_completo}, PIS/PASEP {_random_digits(3)}.{_random_digits(5)}.{_random_digits(2)}-{random.randint(0,9)}, CPF {cpf}. Solicito informações sobre meu FGTS.",
            f"PIS: {_random_digits(3)}.{_random_digits(5)}.{_random_digits(2)}-{random.randint(0,9)}. Solicito informações sobre meu PIS.",
            f"PIS/PASEP {_random_digits(3)}.{_random_digits(5)}.{_random_digits(2)}-{random.randint(0,9)}. Solicito atualização cadastral.",
            f"Meu nome é {nome_completo}, CNH {_random_digits(11)}, CPF {cpf}. Solicito informações sobre multas de trânsito.",
        ]
        out.append(_pick(templates))
    return out


def _negativos_hard(n: int, amostra_texts: List[str]) -> List[str]:
    out: List[str] = []

    base_templates = [
        "Solicito acesso integral aos autos do Processo SEI {sei}, com urgência.",
        "Por qual motivo o Processo SEI nº {sei2} não foi atendido até o presente momento?",
        "Solicito ocorrência atendida pelo PMDF de nº {ocorr}.",
        "Solicito ocorrência atendida pelo CBMDF de nº {ocorr}.",
        "Solicito informações sobre a frota oficial e manutenção.",
        "Solicito cópia do contrato nº {contrato}/{ano} firmado entre órgãos do DF.",
        "Solicito informações sobre a licitação nº {licit}/{ano}.",
        "Gostaria de saber quantos contratos foram firmados em {ano}.",
        "Peço informações sobre o calendário de audiências públicas.",
        "Protocolo: {proto11}. Solicito andamento do processo.",
        "Nota fiscal: {proto11}. Solicito dados consolidados.",
        "Número Interno {proto11}. Solicito esclarecimentos.",
        "ID do contrato: {id8}. Solicito aditivos.",
        "Contato do setor: 99999-9999 (ramal), para dúvidas gerais.",
        "Matrícula {mat8}. Solicito registros de frequência (consolidado).",
        "Inscrição {insc}. Solicito histórico de consumo (sem dados pessoais).",
        "IPTU nº {iptu}. Solicito planta do imóvel (sem dados pessoais).",
        "Documento/Empenho Nº {empenho}. Solicito confirmação de recebimento.",
        "CNPJ do órgão: 12.345.678/0001-90 (informação institucional).",
        "Data do evento: 01/01/2026. Solicito agenda.",
    ]

    for _ in range(n):
        sei = f"{random.randint(0, 99999):05d}-{random.randint(0, 99999999):08d}/{random.randint(2000, 2026)}-{random.randint(0, 99):02d}"
        sei2 = f"{random.randint(0, 99999):05d}-{random.randint(0, 99999999):08d}/{random.randint(2000, 2026)}-{random.randint(0, 99):02d}"
        ocorr = str(random.randint(2000000000000000, 2026999999999999))
        contrato = str(random.randint(10000, 99999))
        ano = str(random.randint(2018, 2026))
        licit = str(random.randint(1, 120))
        proto11 = gerar_cpf_invalido_11dig()
        id8 = str(random.randint(10000000, 99999999))
        mat8 = str(random.randint(10000000, 99999999))
        insc = f"{random.randint(1000, 999999)}-{random.randint(0,9)}"
        iptu = str(random.randint(100000000, 999999999))
        empenho = f"{random.randint(2018, 2026)}NE{random.randint(10000, 99999)}"

        t = _pick(base_templates).format(
            sei=sei,
            sei2=sei2,
            ocorr=ocorr,
            contrato=contrato,
            ano=ano,
            licit=licit,
            proto11=proto11,
            id8=id8,
            mat8=mat8,
            insc=insc,
            iptu=iptu,
            empenho=empenho,
        )
        out.append(t)

    # incorporar “linguagem” da amostra (sem rótulo) como negativos, após scrub
    if amostra_texts:
        for _ in range(min(n, len(amostra_texts))):
            t = scrub_pii(_pick(amostra_texts))
            # reforçar que não é CPF: remove token literal "CPF" quando sobrou
            t = re.sub(r"\bCPF\b", "documento", t, flags=re.IGNORECASE)
            out.append(t)

    return out[: n + min(n, len(amostra_texts))]


def build_dataset(
    dataset_in: str,
    amostra_in: str,
    dataset_out: str,
    seed: int,
    extra_pos: int = 8000,
    extra_neg: int = 8000,
) -> None:
    _seed_everything(seed)

    df_base = pd.read_csv(dataset_in)
    if "pedido" not in df_base.columns or "contem_dados_pessoais" not in df_base.columns:
        raise ValueError("Dataset de entrada precisa ter colunas: pedido, contem_dados_pessoais")

    amostra_texts = _read_amostra_texts(amostra_in)

    novos_pos = _positivos_sinteticos(extra_pos)
    novos_neg = _negativos_hard(extra_neg, amostra_texts)

    df_extra = pd.DataFrame(
        {
            "pedido": novos_pos + novos_neg,
            "contem_dados_pessoais": [1] * len(novos_pos) + [0] * len(novos_neg),
        }
    )

    df = pd.concat([df_base, df_extra], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(dataset_out), exist_ok=True)
    df.to_csv(dataset_out, index=False)


def _make_model(seed: int) -> Pipeline:
    # dois vetorizadores em paralelo: word + char_wb
    feats = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    lowercase=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    lowercase=True,
                    strip_accents="unicode",
                ),
            ),
        ]
    )

    clf = LogisticRegression(
        solver="saga",
        max_iter=3000,
        random_state=seed,
        class_weight="balanced",
    )

    return Pipeline([("features", feats), ("clf", clf)])


def _best_threshold_for_f1(y_true: np.ndarray, proba_pos: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (proba_pos >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


def train_and_export(
    dataset_path: str,
    model_out: str,
    threshold_out: str,
    metrics_out: str,
    seed: int,
) -> None:
    _seed_everything(seed)
    df = pd.read_csv(dataset_path)
    X = df["pedido"].astype(str).fillna("").values
    y = df["contem_dados_pessoais"].astype(int).values

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
    )

    model = _make_model(seed)
    model.fit(X_train, y_train)

    proba_val = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = _best_threshold_for_f1(y_val, proba_val)

    proba_test = model.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= best_t).astype(int)

    metrics = {
        "modelo": "TFIDF+LogReg",
        "seed": seed,
        "dataset": {
            "path": dataset_path,
            "total_amostras": int(len(df)),
            "treino": int(len(X_train)),
            "validacao": int(len(X_val)),
            "teste": int(len(X_test)),
        },
        "threshold": {"best": best_t, "f1_validacao": best_f1},
        "metricas_teste": {
            "f1": float(f1_score(y_test, y_pred)),
            "precisao": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "acuracia": float(accuracy_score(y_test, y_pred)),
            "auc": float(roc_auc_score(y_test, proba_test)),
        },
        "matriz_confusao": {},
        "data_treinamento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics["matriz_confusao"] = {"VP": int(tp), "FP": int(fp), "VN": int(tn), "FN": int(fn)}

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)

    with open(threshold_out, "w", encoding="utf-8") as f:
        json.dump({"threshold": best_t}, f, ensure_ascii=False, indent=2)

    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino/Geração de dataset (Hackathon CGDF)")
    parser.add_argument("--build-dataset", action="store_true", help="Gera dataset ampliado e salva em data/")
    parser.add_argument("--train", action="store_true", help="Treina modelo TF‑IDF e exporta em models/")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dataset-in", type=str, default=os.path.join(_project_root(), "data", "dataset_pedidos_lai.csv"))
    parser.add_argument("--amostra-in", type=str, default=os.path.join(_project_root(), "AMOSTRA_e-SIC - Amostra - SIC.csv"))
    parser.add_argument("--dataset-out", type=str, default=os.path.join(_project_root(), "data", "dataset_pedidos_lai_aug.csv"))

    parser.add_argument("--model-out", type=str, default=os.path.join(_project_root(), "models", "modelo_tfidf.pkl"))
    parser.add_argument("--threshold-out", type=str, default=os.path.join(_project_root(), "models", "threshold.json"))
    parser.add_argument("--metrics-out", type=str, default=os.path.join(_project_root(), "models", "metricas_tfidf.json"))

    parser.add_argument("--extra-pos", type=int, default=8000)
    parser.add_argument("--extra-neg", type=int, default=8000)

    args = parser.parse_args()

    if not args.build_dataset and not args.train:
        parser.error("Informe --build-dataset e/ou --train")

    if args.build_dataset:
        build_dataset(
            dataset_in=args.dataset_in,
            amostra_in=args.amostra_in,
            dataset_out=args.dataset_out,
            seed=args.seed,
            extra_pos=args.extra_pos,
            extra_neg=args.extra_neg,
        )
        print(f"OK: dataset gerado em: {args.dataset_out}")

    if args.train:
        # treina preferencialmente no dataset ampliado, se existir
        dataset_for_train = args.dataset_out if os.path.exists(args.dataset_out) else args.dataset_in
        train_and_export(
            dataset_path=dataset_for_train,
            model_out=args.model_out,
            threshold_out=args.threshold_out,
            metrics_out=args.metrics_out,
            seed=args.seed,
        )
        print(f"OK: modelo exportado em: {args.model_out}")
        print(f"OK: threshold exportado em: {args.threshold_out}")
        print(f"OK: métricas exportadas em: {args.metrics_out}")


if __name__ == "__main__":
    main()


