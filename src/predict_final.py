#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
HACKATHON CGDF - 1º Hackathon em Controle Social
CATEGORIA: Acesso à Informação

Modelo de Detecção de Dados Pessoais em Pedidos LAI
Versão Final - Otimizada para Produção
================================================================================

Este script classifica pedidos de acesso à informação (LAI) quanto à presença
de dados pessoais que deveriam ser classificados como não públicos.

Autor: Solução desenvolvida para o Desafio Participa DF
Data: Janeiro 2026
Versão: 1.0.0
================================================================================
"""

import argparse
import json
import os
import sys

import joblib

# Permitir execução direta e import local do src/
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from features_pii import detectar_pii  # noqa: E402

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "modelo_tfidf.pkl")
THRESHOLD_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "threshold.json")

# ============================================================================
# FUNÇÕES DE EXTRAÇÃO DE FEATURES
# ============================================================================

def _carregar_threshold(path: str) -> float:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("threshold", 0.5))
    except FileNotFoundError:
        return 0.5

# ============================================================================
# FUNÇÃO DE CLASSIFICAÇÃO
# ============================================================================

def classificar_pedido(texto, modelo_path=None):
    """
    Classifica um pedido LAI como contendo ou não dados pessoais.
    
    Parâmetros:
    -----------
    texto : str
        Texto do pedido LAI a ser classificado
    modelo_path : str, opcional
        Caminho para o arquivo do modelo treinado
    
    Retorna:
    --------
    dict
        Dicionário com o resultado da classificação
    """
    if modelo_path is None:
        modelo_path = MODEL_PATH

    modelo = joblib.load(modelo_path)
    threshold = _carregar_threshold(THRESHOLD_PATH)

    t = texto or ""
    proba = modelo.predict_proba([t])[0]
    proba_sem = float(proba[0])
    proba_com = float(proba[1])
    pred = bool(proba_com >= threshold)

    # Explicabilidade (tipos encontrados) — independente da decisão do modelo
    findings = detectar_pii(t)
    tipos_encontrados = findings.tipos
    total_padroes = int(sum(findings.contagens.values()))
    total_tipos = int(len(tipos_encontrados))

    confianca = proba_com if pred else proba_sem

    return {
        "contem_dados_pessoais": pred,
        "confianca": float(confianca),
        "probabilidade_sem_dados": proba_sem,
        "probabilidade_com_dados": proba_com,
        "tipos_dados_encontrados": tipos_encontrados,
        "total_padroes": total_padroes,
        "total_tipos": total_tipos,
        "threshold": float(threshold),
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classifica pedidos LAI quanto à presença de dados pessoais',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python predict_final.py "Texto do pedido LAI"
  python predict_final.py "Meu CPF é 123.456.789-01" --json
  python predict_final.py "Solicito informações..." --modelo /caminho/do/modelo.pkl
        """
    )
    parser.add_argument('texto', type=str, help='Texto do pedido LAI a ser classificado')
    parser.add_argument('--modelo', type=str, help='Caminho para o modelo (opcional)')
    parser.add_argument('--json', action='store_true', help='Retornar resultado em JSON')
    
    args = parser.parse_args()
    
    try:
        resultado = classificar_pedido(args.texto, args.modelo)
        
        if args.json:
            print(json.dumps(resultado, indent=2, ensure_ascii=False))
        else:
            print("=" * 70)
            print("RESULTADO DA CLASSIFICAÇÃO")
            print("=" * 70)
            # Evitar caracteres que quebram em consoles Windows (encoding)
            print(f"Contém dados pessoais: {'SIM' if resultado['contem_dados_pessoais'] else 'NAO'}")
            print(f"Confiança: {resultado['confianca']:.2%}")
            print(f"Probabilidade (Sem dados): {resultado['probabilidade_sem_dados']:.2%}")
            print(f"Probabilidade (Com dados): {resultado['probabilidade_com_dados']:.2%}")
            if resultado['tipos_dados_encontrados']:
                print(f"Tipos de dados encontrados: {', '.join(resultado['tipos_dados_encontrados'])}")
            print(f"Total de padrões detectados: {resultado['total_padroes']}")
            print("=" * 70)
            
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)
