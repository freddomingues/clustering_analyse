from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

def encontrar_k_otimo(df_padronizado, max_k=10):
    """
    Calcula a inércia (WCSS) e o coeficiente de silhueta para um range de K.

    Args:
        df_padronizado (pd.DataFrame): DataFrame com dados padronizados.
        max_k (int): Número máximo de clusters a serem testados.

    Returns:
        dict: Dicionário contendo listas de inércias e scores de silhueta.
    """
    # Correção: Inicializando as listas como vazias.
    inercias = []
    scores_silhueta = []
    range_k = range(2, max_k + 1)
    
    for k in range_k:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(df_padronizado)
        inercias.append(kmeans.inertia_)
        # Calcula o score de silhueta para as labels do k-means atual
        score = silhouette_score(df_padronizado, kmeans.labels_)
        scores_silhueta.append(score)
        
    print("Cálculo de inércia e scores de silhueta concluído.")
    return {'range_k': list(range_k), 'inercias': inercias, 'scores_silhueta': scores_silhueta}

def aplicar_kmeans(df_padronizado, n_clusters=4):
    """
    Aplica o algoritmo K-Means.

    Args:
        df_padronizado (pd.DataFrame): DataFrame com dados padronizados.
        n_clusters (int): Número de clusters.

    Returns:
        np.array: Rótulos dos clusters para cada ponto de dado.
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_padronizado)
    print(f"K-Means aplicado com {n_clusters} clusters.")
    return labels

def aplicar_cluster_hierarquico(df_padronizado, n_clusters=4):
    """
    Aplica o algoritmo de Clusterização Hierárquica Aglomerativa.

    Args:
        df_padronizado (pd.DataFrame): DataFrame com dados padronizados.
        n_clusters (int): Número de clusters.

    Returns:
        np.array: Rótulos dos clusters.
        object: O modelo treinado para gerar o dendrograma.
    """
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agg_clustering.fit_predict(df_padronizado)
    print(f"Clusterização Hierárquica aplicada com {n_clusters} clusters.")
    return labels, agg_clustering

def aplicar_dbscan(df_padronizado, eps=0.5, min_samples=5):
    """
    Aplica o algoritmo DBSCAN.

    Args:
        df_padronizado (pd.DataFrame): DataFrame com dados padronizados.
        eps (float): Raio da vizinhança.
        min_samples (int): Número mínimo de amostras para formar um cluster.

    Returns:
        np.array: Rótulos dos clusters. O rótulo -1 indica ruído.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(df_padronizado)
    # Calcula o número de clusters encontrados, ignorando o ruído (-1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"DBSCAN aplicado com eps={eps} e min_samples={min_samples}.")
    print(f"Número de clusters encontrados: {n_clusters}")
    print(f"Número de pontos de ruído: {n_noise}")
    return labels
