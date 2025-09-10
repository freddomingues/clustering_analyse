# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def gerar_dados_sinteticos(n_clientes=30000, seed=42):
    """
    Gera uma base de dados sintética de clientes inadimplentes,
    incluindo dados sociodemográficos e de comportamento de crédito.
    --- VERSÃO COM MAIOR VARIABILIDADE E REALISMO ---
    """
    print(f"Iniciando a geração de {n_clientes} registros de dados sintéticos...")
    np.random.seed(seed)
    
    # --- DADOS DEMOGRÁFICOS ---
    cliente_id = np.arange(1, n_clientes + 1)
    
    # --- ALTERADO: Idade com distribuição Normal ---
    # Em vez de uma chance igual para todas as idades (uniforme), usamos uma
    # distribuição normal (curva de sino) centrada em 45 anos. Mais realista.
    idade = np.random.normal(loc=45, scale=15, size=n_clientes)
    idade = np.clip(idade, 18, 85).astype(int) # Garante que as idades fiquem entre 18 e 85

    sexo = np.random.choice(['Masculino', 'Feminino'], size=n_clientes, p=[0.48, 0.52]) # Proporção levemente ajustada
    estado_civil = np.random.choice(['Solteiro', 'Casado', 'Divorciado', 'Viúvo'], size=n_clientes, p=[0.35, 0.45, 0.15, 0.05])

    # --- ALTERADO: Dependentes com distribuição de Poisson ---
    # A distribuição de Poisson é ideal para dados de contagem. A maioria das pessoas
    # terá 0, 1 ou 2 dependentes, e poucos terão mais que isso.
    numero_dependentes = np.random.poisson(lam=1.2, size=n_clientes)
    numero_dependentes = np.clip(numero_dependentes, 0, 8) # Evita valores extremos raros
    
    nivel_educacional = np.random.choice(
        ['Fundamental', 'Médio', 'Superior', 'Pós-graduação'], 
        size=n_clientes, p=[0.15, 0.50, 0.25, 0.10]
    )
    tipo_emprego = np.random.choice(
        ['CLT', 'Autônomo', 'Funcionário Público', 'Empresário', 'Desempregado'],
        size=n_clientes, p=[0.50, 0.20, 0.10, 0.10, 0.10]
    )

    # --- DADOS FINANCEIROS E DE DÍVIDA ---
    base_renda = {'Fundamental': 1800, 'Médio': 3500, 'Superior': 7000, 'Pós-graduação': 12000}
    modificador_emprego = {'CLT': 1.0, 'Autônomo': 1.2, 'Funcionário Público': 1.3, 'Empresário': 1.8, 'Desempregado': 0.3}
    renda_mensal = [base_renda[edu] * modificador_emprego[emp] * np.random.uniform(0.7, 1.3) for edu, emp in zip(nivel_educacional, tipo_emprego)]
    renda_mensal = np.array(renda_mensal).round(2)

    # --- ALTERADO: Histórico de pagamento com mais variabilidade ---
    # Criamos dois perfis: "bons pagadores" e "pagadores de risco" para que a variável
    # não seja sempre alta, criando mais contraste.
    risky_mask = np.random.rand(n_clientes) < 0.3 # 30% são mais arriscados
    good_payer_mask = ~risky_mask
    historico_pagamento_recente = np.zeros(n_clientes)
    historico_pagamento_recente[good_payer_mask] = np.random.beta(a=8, b=2, size=good_payer_mask.sum()) # Tendência a pagar em dia
    historico_pagamento_recente[risky_mask] = np.random.beta(a=2, b=3, size=risky_mask.sum()) # Tendência a atrasar
    historico_pagamento_recente = historico_pagamento_recente.round(2)
    
    score_credito = 300 + (renda_mensal / 200) + (idade * 1.5) + (historico_pagamento_recente * 300)
    score_credito += np.random.randint(-50, 50, size=n_clientes)
    score_credito[tipo_emprego == 'Desempregado'] -= 100
    score_credito = np.clip(score_credito, 300, 950).astype(int)

    produto_origem_divida = np.random.choice(
        ['Cartão de Crédito', 'Empréstimo Pessoal', 'Financiamento Veículo', 'Cheque Especial'],
        size=n_clientes, p=[0.4, 0.3, 0.15, 0.15]
    )

    # --- ALTERADO: Tempo de débito com distribuição Exponencial ---
    # A maioria das dívidas será mais recente, com poucas sendo muito antigas.
    # A distribuição exponencial modela isso muito bem.
    tempo_de_debito_meses = np.random.exponential(scale=18, size=n_clientes) # Média de 18 meses
    tempo_de_debito_meses = np.clip(tempo_de_debito_meses, 1, 60).astype(int)
    
    valor_divida = (renda_mensal * np.random.uniform(0.2, 2.0, size=n_clientes))
    valor_divida[valor_divida < 100] = 100
    
    # --- CRIAÇÃO DO DATAFRAME FINAL ---
    df = pd.DataFrame({
        'cliente_id': cliente_id, 
        'idade': idade, 
        'sexo': sexo,
        'estado_civil': estado_civil, 
        'nivel_educacional': nivel_educacional,
        'numero_dependentes': numero_dependentes, 
        'tipo_emprego': tipo_emprego,
        'renda_mensal': renda_mensal, 
        'score_credito': score_credito,
        'historico_pagamento_recente': historico_pagamento_recente,
        'produto_origem_divida': produto_origem_divida,
        'tempo_de_debito_meses': tempo_de_debito_meses,
        'valor_divida': valor_divida.round(2)
    })
    
    print("Base de dados sintética (com maior variabilidade) gerada com sucesso.")
    print(f"Dimensões do DataFrame: {df.shape}")
    return df