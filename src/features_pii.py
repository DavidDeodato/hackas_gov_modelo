#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilitários de detecção explicável de padrões de dados pessoais.

IMPORTANTE:
- Aqui a detecção é usada para **explicabilidade** (tipos encontrados) e para
  ajudar na geração de dataset sintético/hard negatives.
- A decisão final do classificador no pipeline TF‑IDF fica no modelo treinado.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PIIFindings:
    tipos: List[str]
    contagens: Dict[str, int]


_RE_CPF_FORMATADO = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")
_RE_CPF_11_DIGITOS = re.compile(r"\b\d{11}\b")
_RE_CPF_ESPACADO = re.compile(r"\b\d{3}\s+\d{3}\s+\d{3}\s+\d{2}\b")

_RE_RG_FORMATADO = re.compile(r"\b\d{1,2}\.\d{3}\.\d{3}-[\dXx]\b")
_RE_TEL_PAREN = re.compile(r"\(\d{2}\)\s*\d{4,5}-\d{4}")
_RE_TEL_HIFEN = re.compile(r"\b\d{4,5}-\d{4}\b")
_RE_TEL_ESPACOS = re.compile(r"\b\d{2}\s+\d{4,5}\s+\d{4}\b")

_RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_RE_CEP = re.compile(r"\b\d{5}-\d{3}\b")
_RE_DATA = re.compile(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b")

_RE_PIS = re.compile(r"\b\d{3}\.\d{5}\.\d{2}-\d\b")
_RE_TITULO = re.compile(r"\b\d{12}\b")


def _normalize_email_obfuscation(texto: str) -> str:
    # formas comuns em PT-BR
    texto = re.sub(r"\s*\[\s*arroba\s*\]\s*", "@", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\s*\(\s*at\s*\)\s*", "@", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\s+\bat\b\s+", "@", texto, flags=re.IGNORECASE)
    return texto


def detectar_pii(texto: str) -> PIIFindings:
    t = texto or ""
    t_norm_email = _normalize_email_obfuscation(t)
    t_lower = t.lower()

    contagens: Dict[str, int] = {}

    contagens["cpf_formatado"] = len(_RE_CPF_FORMATADO.findall(t))
    contagens["cpf_espacado"] = len(_RE_CPF_ESPACADO.findall(t))
    # 11 dígitos é ambíguo (protocolo/ID), então só conta como CPF “forte” se houver contexto
    cpf_11 = _RE_CPF_11_DIGITOS.findall(t)
    contagens["cpf_11_contexto"] = sum(1 for x in cpf_11 if "cpf" in t_lower)

    contagens["rg_formatado"] = len(_RE_RG_FORMATADO.findall(t))
    contagens["tel_paren"] = len(_RE_TEL_PAREN.findall(t))
    contagens["tel_hifen"] = len(_RE_TEL_HIFEN.findall(t))
    contagens["tel_espacos"] = len(_RE_TEL_ESPACOS.findall(t))
    contagens["email"] = len(_RE_EMAIL.findall(t_norm_email))
    contagens["cep"] = len(_RE_CEP.findall(t))
    contagens["data"] = len(_RE_DATA.findall(t))
    contagens["pis"] = len(_RE_PIS.findall(t))
    contagens["titulo_12_contexto"] = len(_RE_TITULO.findall(t)) if ("título" in t_lower or "titulo" in t_lower) else 0

    tipos: List[str] = []
    if contagens["cpf_formatado"] or contagens["cpf_espacado"] or contagens["cpf_11_contexto"]:
        tipos.append("CPF")
    if contagens["rg_formatado"]:
        tipos.append("RG")
    if contagens["tel_paren"] or contagens["tel_hifen"] or contagens["tel_espacos"]:
        tipos.append("Telefone")
    if contagens["email"]:
        tipos.append("E-mail")
    if contagens["cep"]:
        tipos.append("CEP/Endereço")
    if contagens["data"]:
        tipos.append("Data")
    if contagens["pis"]:
        tipos.append("PIS/PASEP")
    if contagens["titulo_12_contexto"]:
        tipos.append("Título de Eleitor")

    return PIIFindings(tipos=tipos, contagens=contagens)


def scrub_pii(texto: str) -> str:
    """Remove/substitui padrões óbvios para evitar vazamento quando usamos textos como base."""
    t = texto or ""
    t = _RE_CPF_FORMATADO.sub("[CPF]", t)
    t = _RE_CPF_ESPACADO.sub("[CPF]", t)
    t = _RE_RG_FORMATADO.sub("[RG]", t)
    t = _RE_PIS.sub("[PIS]", t)
    t = _RE_TEL_PAREN.sub("[TEL]", t)
    t = _RE_TEL_HIFEN.sub("[TEL]", t)
    t = _RE_TEL_ESPACOS.sub("[TEL]", t)
    t = _RE_EMAIL.sub("[EMAIL]", _normalize_email_obfuscation(t))
    t = _RE_CEP.sub("[CEP]", t)
    t = _RE_DATA.sub("[DATA]", t)
    return t


