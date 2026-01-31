# test_frases.py
# -*- coding: utf-8 -*-

import os
import sys

# Garantir import do módulo src/
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.predict_final import classificar_pedido


def build_cases():
    cases = []

    # --------------------------
    # POSITIVOS (padrões claros)
    # --------------------------
    cases += [
        ("Meu CPF é 123.456.789-01. Solicito informações sobre meu processo.", True),
        ("CPF: 11122233344. Peço acesso ao meu cadastro.", True),
        ("RG 12.345.678-9 e CPF 987.654.321-00. Solicito cópia do meu pedido.", True),
        ("Contato: (61) 91234-5678. Preciso atualizar meus dados.", True),
        ("Telefone 91234-5678. Solicito informações.", True),
        ("E-mail: joao.silva@gmail.com. Solicito acesso às minhas informações.", True),
        ("Meu email é maria+teste@empresa.com.br. Quero retificar meus dados.", True),
        ("Endereço: SQN 112 Bloco A, Brasília-DF. CEP 70000-000.", True),
        ("CEP 72123456 e endereço atualizado. Solicito alteração cadastral.", True),
        ("Data de nascimento: 01/01/1990. Solicito informações do meu cadastro.", True),
        ("Nascido em 25-12-1988. Peço acesso às informações.", True),
        ("CNH: 98765432100. Solicito correção no sistema.", True),
        ("PIS: 123.45678.90-1. Solicito informações sobre benefício.", True),
        ("CNPJ 12.345.678/0001-90. Solicito informação sobre cadastro da empresa.", False),
        ("Título de eleitor: 123456789012. Solicito regularização.", True),
        ("Meu CPF é ***.456.789-01. Solicito anonimização.", True),
    ]

    # --------------------------
    # NEGATIVOS (pedidos típicos)
    # --------------------------
    cases += [
        ("Solicito cópia do contrato nº 20529/2026 firmado pela secretaria.", False),
        ("Gostaria de saber quantos contratos foram firmados em 2023.", False),
        ("Peço acesso ao edital e termo de referência da licitação nº 30/2022.", False),
        ("Solicito relatório de execução orçamentária de 2024.", False),
        ("Quais foram os gastos com cartão corporativo no ano de 2025?", False),
        ("Solicito lista de obras em execução e seus prazos.", False),
        ("Peço dados estatísticos do programa de combate à dengue.", False),
        ("Solicito informações sobre a frota oficial e manutenção.", False),
        ("Quais os valores pagos em diárias no mês de janeiro de 2026?", False),
        ("Solicito ata de julgamento da licitação nº 98/2020.", False),
        ("Gostaria de conhecer o planejamento estratégico da secretaria.", False),
        ("Peço o organograma atualizado do órgão.", False),
        ("Solicito quantidade de servidores por cargo (consolidado).", False),
        ("Quais contratos ativos têm valores superiores a R$ 100.000,00?", False),
        ("Solicito relação de convênios celebrados em 2022.", False),
        ("Peço informações sobre o calendário de audiências públicas.", False),
    ]

    # ---------------------------------------------------
    # EDGE CASES: positivos que o modelo tende a ERRAR
    # (dados pessoais sem padrões numéricos explícitos)
    # ---------------------------------------------------
    cases += [
        ("O nome completo do servidor é João Carlos da Silva, matrícula 45872.", True),  # nome é dado pessoal
        ("Sou Ana Paula Pereira e gostaria de informações sobre meu processo.", True),   # só nome
        ("Me chamo Rodrigo Ferreira Campos. Solicito informações.", True),              # só nome
        ("Meu nome: Carla Mendes. Quero acesso às minhas informações.", True),          # nome
        ("Gostaria de corrigir meu nome no cadastro: Beatriz Souza Lima.", True),       # nome
        ("Solicito a exclusão dos meus dados pessoais. Nome: Lucas Andrade.", True),    # nome + contexto
        ("Minha mãe se chama Maria de Fátima. Preciso de informações do benefício.", True),  # nome
        ("Sou o cidadão Pedro Henrique e solicito acesso ao meu requerimento.", True),  # 'sou o' (keyword), mas sem número
        ("Sou a cidadã Juliana. Solicito informações.", True),
        ("Meu endereço é Rua das Flores, Brasília-DF. Preciso atualizar cadastro.", True),  # endereço sem CEP
        ("Endereço: Avenida Central, nº 120. Solicito atualização.", True),                 # endereço sem CEP
        ("Contato: joao [arroba] email.com. Solicito correção.", True),                     # email ofuscado
        ("Meu e-mail é joao (at) dominio.com. Solicito acesso.", True),                     # email ofuscado
        ("Telefone: 61 91234 5678. Solicito retorno.", True),                               # telefone sem hífen/parênteses
        ("RG: 12345678X. Solicito atualização cadastral.", True),                           # RG com letra sem formato previsto
        ("CPF: 123 456 789 01. Solicito informações.", True),                               # CPF espaçado
    ]

    # ---------------------------------------------------
    # EDGE CASES: negativos que o modelo tende a MARCAR como positivo
    # (números longos/protocolos confundidos com CPF/CNH)
    # ---------------------------------------------------
    cases += [
        ("Protocolo: 12345678901. Solicito andamento do processo.", False),  # 11 dígitos (vai virar CPF/CNH)
        ("Número do processo: 202612345678. Solicito cópia integral.", False),  # 12 dígitos (título)
        ("Nota fiscal: 12345678901. Solicito dados consolidados.", False),     # 11 dígitos
        ("ID do contrato: 12345678. Solicito aditivos.", False),               # 8 dígitos (CEP pode bater se sem borda)
        ("Código interno 70000000 para referência orçamentária.", False),       # 8 dígitos (CEP)
        ("Série do equipamento: 123456789012. Solicito inventário.", False),   # 12 dígitos
        ("Chave do documento: 123.45678.90-1 (formato técnico).", False),      # ambíguo sem contexto (evitar FP)
        ("Contato do setor: 99999-9999 (ramal), para dúvidas gerais.", False), # pode parecer telefone
        ("Data do evento: 01/01/2026. Solicito agenda.", False),               # data não é necessariamente dado pessoal
        ("CNPJ do órgão: 12.345.678/0001-90 (informação institucional).", False),  # CNPJ pode ser institucional
    ]

    # ---------------------------------------------------
    # Completar até 100 com variações determinísticas
    # ---------------------------------------------------
    templates_pos = [
        ("Peço anonimização. CPF {cpf}.", True),
        ("Solicito correção cadastral. Email {email}.", True),
        ("Atualizar contato: (61) {tel}.", True),
        ("Dados: RG {rg}. Solicito acesso.", True),
        ("Endereço e CEP {cep}. Solicito alteração.", True),
    ]
    templates_neg = [
        ("Solicito informações sobre a licitação nº {num}/2024.", False),
        ("Peço cópia do contrato nº {num}/2026.", False),
        ("Quero relatório de despesas de {ano}.", False),
        ("Solicito dados estatísticos do programa {prog} em {ano}.", False),
        ("Peço organograma do órgão para {ano}.", False),
    ]
    cpfs = ["050.367.184-37", "611.740.591-09", "190.020.173-91", "232.706.746-43", "536.272.930-61"]
    emails = ["lucas@yahoo.com.br", "vanessa@uol.com.br", "rodrigo@hotmail.com", "bruno@gmail.com", "carolina@bol.com.br"]
    tels = ["91234-5678", "99118-53878"[-9:], "99402-66495"[-9:], "99073-85119"[-9:], "99247-14647"[-9:]]
    rgs = ["83.116.260-8", "29.247.572-9", "13.311.319-9", "72.020.609-9", "76.947.249-1"]
    ceps = ["70000-000", "72494-428", "71452-235", "72924-668", "72650-930"]
    progs = ["MaisMedicos", "CombateDengue", "TransporteEscolar", "Emprego", "Habitação"]
    anos = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]

    i = 0
    while len(cases) < 100:
        # alternar positivo/negativo pra equilibrar
        if len(cases) % 2 == 0:
            tpl, y = templates_pos[i % len(templates_pos)]
            text = tpl.format(
                cpf=cpfs[i % len(cpfs)],
                email=emails[i % len(emails)],
                tel=tels[i % len(tels)],
                rg=rgs[i % len(rgs)],
                cep=ceps[i % len(ceps)],
            )
        else:
            tpl, y = templates_neg[i % len(templates_neg)]
            text = tpl.format(
                num=str(10_000 + i),
                ano=anos[i % len(anos)],
                prog=progs[i % len(progs)],
            )
        cases.append((text, y))
        i += 1

    return cases[:100]


def main():
    cases = build_cases()

    total = len(cases)
    ok = 0
    wrong = []

    for idx, (texto, expected) in enumerate(cases, start=1):
        r = classificar_pedido(texto)
        pred = bool(r["contem_dados_pessoais"])
        if pred == expected:
            ok += 1
        else:
            wrong.append((idx, expected, pred, r.get("tipos_dados_encontrados", []), texto))

    print("=" * 80)
    print("AVALIAÇÃO RÁPIDA COM 100 FRASES (ground truth = interpretação do edital)")
    print("=" * 80)
    print(f"Acertos: {ok}/{total} = {ok/total:.1%}")
    print(f"Erros:   {len(wrong)}/{total} = {len(wrong)/total:.1%}")
    print()

    if wrong:
        print("Casos divergentes (onde o modelo provavelmente não generaliza bem):")
        for (idx, exp, pred, tipos, texto) in wrong[:30]:
            print("-" * 80)
            print(f"[{idx:03d}] esperado={exp}  previsto={pred}  tipos_detectados={tipos}")
            print(texto)
        if len(wrong) > 30:
            print("-" * 80)
            print(f"... mais {len(wrong)-30} divergências omitidas")
    else:
        print("Nenhuma divergência nas 100 frases (muito improvável fora do sintético).")


if __name__ == "__main__":
    main()