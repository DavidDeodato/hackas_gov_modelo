#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
HACKATHON CGDF - 1º Hackathon em Controle Social
CATEGORIA: Acesso à Informação

Modelo de Detecção de Dados Pessoais em Pedidos LAI
================================================================================

Este script classifica pedidos de acesso à informação (LAI) quanto à presença
de dados pessoais que deveriam ser classificados como não públicos.

Autor: Solução desenvolvida para o Desafio Participa DF
Data: Janeiro 2026
================================================================================
"""

import pickle
import re
import pandas as pd
import argparse
import json
import sys
import os

# ============================================================================
# FUNÇÕES DE EXTRAÇÃO DE FEATURES
# ============================================================================

def extrair_features_regex(texto):
    """Extrai features baseadas em regex para dados pessoais"""
    features = {}
    
    # Contagem de padrões
    features['count_cpf'] = len(re.findall(r'\d{3}\.\d{3}\.\d{3}-\d{2}', texto))
    features['count_cpf_numeros'] = len(re.findall(r'\b\d{11}\b', texto))
    features['count_rg'] = len(re.findall(r'\d{1,2}\.\d{3}\.\d{3}-\d{1}', texto))
    features['count_telefone'] = len(re.findall(r'\(\d{2}\)\s*\d{4,5}-\d{4}', texto))
    features['count_email'] = len(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', texto))
    features['count_data'] = len(re.findall(r'\d{2}[/-]\d{2}[/-]\d{4}', texto))
    features['count_cep'] = len(re.findall(r'\d{5}-?\d{3}', texto))
    
    # Features binárias
    features['has_cpf'] = 1 if features['count_cpf'] > 0 or features['count_cpf_numeros'] > 0 else 0
    features['has_rg'] = 1 if features['count_rg'] > 0 else 0
    features['has_telefone'] = 1 if features['count_telefone'] > 0 else 0
    features['has_email'] = 1 if features['count_email'] > 0 else 0
    features['has_data'] = 1 if features['count_data'] > 0 else 0
    features['has_cep'] = 1 if features['count_cep'] > 0 else 0
    
    # Features agregadas
    features['total_padroes'] = (features['count_cpf'] + features['count_cpf_numeros'] + 
                                  features['count_rg'] + features['count_telefone'] + 
                                  features['count_email'] + features['count_data'] + 
                                  features['count_cep'])
    features['total_tipos'] = (features['has_cpf'] + features['has_rg'] + 
                                features['has_telefone'] + features['has_email'] + 
                                features['has_data'] + features['has_cep'])
    
    return features

def extrair_features_linguisticas(texto):
    """Extrai features linguísticas do texto"""
    features = {}
    
    # Comprimento
    features['len_texto'] = len(texto)
    features['len_words'] = len(texto.split())
    features['avg_word_len'] = sum(len(w) for w in texto.split()) / len(texto.split()) if texto.split() else 0
    
    # Caracteres especiais
    features['count_numeros'] = sum(c.isdigit() for c in texto)
    features['count_maiusculas'] = sum(c.isupper() for c in texto)
    features['count_minusculas'] = sum(c.islower() for c in texto)
    features['count_especiais'] = len(texto) - features['count_numeros'] - features['count_maiusculas'] - features['count_minusculas']
    
    # Palavras-chave indicativas de dados pessoais
    palavras_chave = ['nome', 'cpf', 'rg', 'telefone', 'email', 'endereço', 'data', 'nascimento',
                      'cnh', 'título', 'eleitor', 'pis', 'pasep', 'documento', 'identidade']
    texto_lower = texto.lower()
    for palavra in palavras_chave:
        features[f'keyword_{palavra}'] = 1 if palavra in texto_lower else 0
    features['total_keywords'] = sum(features[f'keyword_{p}'] for p in palavras_chave)
    
    return features

# ============================================================================
# FUNÇÃO DE CLASSIFICAÇÃO
# ============================================================================

def classificar_pedido(texto, modelo_path='models/modelo_xgboost.pkl'):
    """
    Classifica um pedido LAI como contendo ou não dados pessoais.
    
    Parâmetros:
    -----------
    texto : str
        Texto do pedido LAI a ser classificado
    modelo_path : str
        Caminho para o arquivo do modelo treinado
    
    Retorna:
    --------
    dict
        Dicionário com o resultado da classificação:
        - contem_dados_pessoais: bool
        - confianca: float
        - probabilidade_classe_0: float
        - probabilidade_classe_1: float
        - tipos_dados_encontrados: list
        - total_padroes: int
        - total_tipos: int
    """
    # Carregar modelo
    if not os.path.isabs(modelo_path):
        # Se for caminho relativo, adicionar diretório do script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        projeto_dir = os.path.dirname(script_dir)
        modelo_path = os.path.join(projeto_dir, modelo_path)
    
    with open(modelo_path, 'rb') as f:
        modelo = pickle.load(f)
    
    # Extrair features
    feat_regex = extrair_features_regex(texto)
    feat_ling = extrair_features_linguisticas(texto)
    features = {**feat_regex, **feat_ling}
    
    # Ordem exata das features do modelo
    feature_cols = ['count_cpf', 'count_cpf_numeros', 'count_rg', 'count_telefone',
                    'count_email', 'count_data', 'count_cep', 'has_cpf', 'has_rg',
                    'has_telefone', 'has_email', 'has_data', 'has_cep', 'total_padroes',
                    'total_tipos', 'len_texto', 'len_words', 'avg_word_len',
                    'count_numeros', 'count_maiusculas', 'count_minusculas',
                    'count_especiais', 'keyword_nome', 'keyword_cpf', 'keyword_rg',
                    'keyword_telefone', 'keyword_email', 'keyword_endereço',
                    'keyword_data', 'keyword_nascimento', 'keyword_cnh',
                    'keyword_título', 'keyword_eleitor', 'keyword_pis', 'keyword_pasep',
                    'keyword_documento', 'keyword_identidade', 'total_keywords']
    
    # Criar DataFrame na ordem correta
    X_input = pd.DataFrame([[features.get(k, 0) for k in feature_cols]], columns=feature_cols)
    
    # Predição
    predicao = modelo.predict(X_input)[0]
    probabilidade = modelo.predict_proba(X_input)[0]
    
    # Detectar tipos de dados encontrados
    tipos_encontrados = []
    if feat_regex['has_cpf']:
        tipos_encontrados.append('CPF')
    if feat_regex['has_rg']:
        tipos_encontrados.append('RG')
    if feat_regex['has_telefone']:
        tipos_encontrados.append('Telefone')
    if feat_regex['has_email']:
        tipos_encontrados.append('E-mail')
    if feat_regex['has_cep']:
        tipos_encontrados.append('CEP/Endereço')
    if feat_regex['has_data']:
        tipos_encontrados.append('Data')
    
    return {
        'contem_dados_pessoais': bool(predicao),
        'confianca': float(max(probabilidade)),
        'probabilidade_classe_0': float(probabilidade[0]),
        'probabilidade_classe_1': float(probabilidade[1]),
        'tipos_dados_encontrados': tipos_encontrados,
        'total_padroes': feat_regex['total_padroes'],
        'total_tipos': feat_regex['total_tipos']
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classifica pedidos LAI quanto à presença de dados pessoais'
    )
    parser.add_argument(
        'texto',
        type=str,
        help='Texto do pedido LAI a ser classificado'
    )
    parser.add_argument(
        '--modelo',
        type=str,
        default='models/modelo_xgboost.pkl',
        help='Caminho para o modelo (padrão: models/modelo_xgboost.pkl)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Retornar resultado em formato JSON'
    )
    
    args = parser.parse_args()
    
    try:
        resultado = classificar_pedido(args.texto, args.modelo)
        
        if args.json:
            print(json.dumps(resultado, indent=2, ensure_ascii=False))
        else:
            print("=" * 70)
            print("RESULTADO DA CLASSIFICAÇÃO")
            print("=" * 70)
            print(f"Contém dados pessoais: {'SIM' if resultado['contem_dados_pessoais'] else 'NÃO'}")
            print(f"Confiança: {resultado['confianca']:.2%}")
            print(f"Probabilidade (Sem dados): {resultado['probabilidade_classe_0']:.2%}")
            print(f"Probabilidade (Com dados): {resultado['probabilidade_classe_1']:.2%}")
            if resultado['tipos_dados_encontrados']:
                print(f"Tipos de dados encontrados: {', '.join(resultado['tipos_dados_encontrados'])}")
            print(f"Total de padrões detectados: {resultado['total_padroes']}")
            print("=" * 70)
            
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)
