import pandas as pd
import numpy as np

def gerar_dados_sinteticos(n_clientes=20000, seed=42):
    """
    Gera uma base de dados sintética de clientes inadimplentes.

    Args:
        n_clientes (int): Número de clientes a serem gerados.
        seed (int): Semente para reprodutibilidade dos resultados.

    Returns:
        pd.DataFrame: DataFrame contendo os dados dos clientes.
    """
    np.random.seed(seed)
    
    # Geração das variáveis
    cliente_id = np.arange(1, n_clientes + 1)
    idade = np.random.randint(18, 81, size=n_clientes)
    sexo = np.random.choice(['M', 'F'], size=n_clientes, p=[0.5, 0.5])
    
    # Renda com distribuição normal, garantindo um valor mínimo
    renda_mensal = np.random.normal(loc=5000, scale=1500, size=n_clientes)
    renda_mensal[renda_mensal < 1000] = 1000
    
    # Valor da dívida correlacionado à renda, mas com variabilidade
    valor_divida = (renda_mensal * np.random.uniform(0.1, 0.8, size=n_clientes)) + np.random.normal(0, 200, size=n_clientes)
    valor_divida[valor_divida < 100] = 100
    
    score_credito = np.random.randint(300, 851, size=n_clientes)
    tempo_de_debito = np.random.randint(1, 61, size=n_clientes)
    
    # Criação do DataFrame
    df = pd.DataFrame({
        'cliente_id': cliente_id,
        'idade': idade,
        'sexo': sexo,
        'renda_mensal': renda_mensal,
        'valor_divida': valor_divida,
        'score_credito': score_credito,
        'tempo_de_debito': tempo_de_debito
    })
    
    print("Base de dados sintética gerada com sucesso.")
    print(f"Dimensões do DataFrame: {df.shape}")
    return df

if __name__ == '__main__':
    # Exemplo de uso: gerar e salvar os dados em um arquivo CSV
    dados_clientes = gerar_dados_sinteticos()
    dados_clientes.to_csv('dados_clientes_inadimplentes.csv', index=False)
    print("\nDados salvos em 'dados_clientes_inadimplentes.csv'")
    print("\nPrimeiras 5 linhas do DataFrame:")
    print(dados_clientes.head())
