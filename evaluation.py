from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd
import numpy as np

def avaliar_modelos(df_padronizado, labels_dict):
    """
    Calcula métricas de avaliação para diferentes resultados de clusterização.

    Args:
        df_padronizado (pd.DataFrame): DataFrame com dados padronizados.
        labels_dict (dict): Dicionário com nomes dos modelos e seus respectivos rótulos.

    Returns:
        pd.DataFrame: DataFrame com as métricas de avaliação para cada modelo.
    """
    # Correção: Inicializando a lista para armazenar os resultados.
    resultados = []
    
    for nome_modelo, labels in labels_dict.items():
        # Ignorar avaliação se houver apenas 1 cluster ou todos os pontos forem ruído.
        # Isso é comum em resultados do DBSCAN com parâmetros mal ajustados.
        if len(set(labels)) < 2:
            print(f"Avaliação pulada para o modelo '{nome_modelo}' pois encontrou menos de 2 clusters.")
            continue
            
        sil_score = silhouette_score(df_padronizado, labels)
        db_score = davies_bouldin_score(df_padronizado, labels)
        
        resultados.append({
            'Modelo': nome_modelo,
            'Coeficiente de Silhueta': sil_score,
            'Índice de Davies-Bouldin': db_score
        })
        
    if not resultados:
        print("Nenhum modelo pôde ser avaliado.")
        return pd.DataFrame()

    df_resultados = pd.DataFrame(resultados)
    print("Avaliação dos modelos concluída.")
    return df_resultados

def analisar_perfis_clusters(df_original, labels, nome_modelo):
    """
    Calcula as médias das variáveis para cada cluster e cria um perfil.

    Args:
        df_original (pd.DataFrame): DataFrame original (não padronizado).
        labels (np.array): Rótulos dos clusters.
        nome_modelo (str): Nome do modelo para adicionar ao DataFrame.

    Returns:
        pd.DataFrame: DataFrame com o perfil médio de cada cluster.
    """
    df_analise = df_original.copy()
    coluna_cluster = f'cluster_{nome_modelo}'
    df_analise[coluna_cluster] = labels
    
    # Excluir ruído (pontos com label -1) da análise de perfil para DBSCAN
    if -1 in df_analise[coluna_cluster].unique():
        df_analise = df_analise[df_analise[coluna_cluster] != -1]

    if df_analise.empty:
        print(f"Análise de perfil para '{nome_modelo}' não pôde ser gerada (sem clusters válidos).")
        return pd.DataFrame()

    perfil_clusters = df_analise.groupby(coluna_cluster).mean()
    perfil_clusters['n_clientes'] = df_analise[coluna_cluster].value_counts()
    
    # Correção: Verificar se a coluna 'id_cliente' existe antes de tentar removê-la.
    if 'id_cliente' in perfil_clusters.columns:
        perfil_clusters = perfil_clusters.drop(columns=['id_cliente'])
    
    print(f"Análise de perfil para o modelo '{nome_modelo}' concluída.")
    return perfil_clusters
