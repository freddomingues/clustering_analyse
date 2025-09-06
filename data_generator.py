# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def gerar_dados_sinteticos(n_clientes=20000, seed=42):
    """
    Gera uma base de dados sintética de clientes inadimplentes,
    incluindo dados sociodemográficos e de comportamento de crédito.

    Args:
        n_clientes (int): Número de clientes a serem gerados.
        seed (int): Semente para reprodutibilidade dos resultados.

    Returns:
        pd.DataFrame: DataFrame contendo os dados detalhados dos clientes.
    """
    print(f"Iniciando a geração de {n_clientes} registros de dados sintéticos...")
    np.random.seed(seed)
    
    # --- DADOS DEMOGRÁFICOS ---
    cliente_id = np.arange(1, n_clientes + 1)
    idade = np.random.randint(18, 81, size=n_clientes)
    # >>> VARIÁVEL 'SEXO' ADICIONADA <<<
    sexo = np.random.choice(['Masculino', 'Feminino'], size=n_clientes)
    estado_civil = np.random.choice(['Solteiro', 'Casado', 'Divorciado', 'Viúvo'], size=n_clientes)
    numero_dependentes = np.random.randint(0, 6, size=n_clientes) # 0 a 5 dependentes
    nivel_educacional = np.random.choice(
        ['Fundamental', 'Médio', 'Superior', 'Pós-graduação'], 
        size=n_clientes
    )
    tipo_emprego = np.random.choice(
        ['CLT', 'Autônomo', 'Funcionário Público', 'Empresário', 'Desempregado'],
        size=n_clientes
    )

    # --- DADOS FINANCEIROS E DE DÍVIDA (COM CORRELAÇÕES) ---
    
    # Renda baseada no nível educacional e tipo de emprego
    base_renda = {'Fundamental': 1800, 'Médio': 3500, 'Superior': 7000, 'Pós-graduação': 12000}
    modificador_emprego = {'CLT': 1.0, 'Autônomo': 1.2, 'Funcionário Público': 1.3, 'Empresário': 1.8, 'Desempregado': 0.3}
    
    renda_mensal = [base_renda[edu] * modificador_emprego[emp] * np.random.uniform(0.7, 1.3) for edu, emp in zip(nivel_educacional, tipo_emprego)]
    renda_mensal = np.array(renda_mensal).round(2)

    # Histórico de pagamento (percentual de contas pagas em dia nos últimos 12 meses)
    historico_pagamento_recente = np.random.beta(a=5, b=2, size=n_clientes).round(2) # Tende a ser mais alto
    
    # Score de crédito influenciado por renda, idade e histórico
    score_credito = 300 + (renda_mensal / 200) + (idade * 1.5) + (historico_pagamento_recente * 300)
    score_credito += np.random.randint(-50, 50, size=n_clientes)
    score_credito[tipo_emprego == 'Desempregado'] -= 100 # Penalidade para desempregados
    score_credito = np.clip(score_credito, 300, 950).astype(int)

    # Dívida
    produto_origem_divida = np.random.choice(
        ['Cartão de Crédito', 'Empréstimo Pessoal', 'Financiamento Veículo', 'Cheque Especial'],
        size=n_clientes,
        p=[0.4, 0.3, 0.15, 0.15]
    )
    tempo_de_debito_meses = np.random.randint(1, 61, size=n_clientes)
    
    # Valor da dívida correlacionado à renda
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
    
    print("Base de dados sintética gerada com sucesso.")
    print(f"Dimensões do DataFrame: {df.shape}")
    return df

if __name__ == '__main__':
    # Exemplo de uso para gerar e salvar os dados em um arquivo Excel
    # Usar Excel (.xlsx) é melhor para compatibilidade
    nome_arquivo_saida = 'base_sintetica_dividas.xlsx'
    dados_clientes = gerar_dados_sinteticos()
    
    # Requer a instalação da biblioteca 'openpyxl' (pip install openpyxl)
    dados_clientes.to_excel(nome_arquivo_saida, index=False)
    
    print(f"\nDados salvos em '{nome_arquivo_saida}'")
    print("\nPrimeiras 5 linhas do DataFrame gerado:")
    print(dados_clientes.head())
    print("\nTipos de dados das colunas:")
    print(dados_clientes.info())