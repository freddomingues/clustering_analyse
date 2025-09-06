# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
import streamlit as st

# >>> OTIMIZAÇÃO: Adicionando cache do Streamlit <<<
# Esta função é a mais demorada. O cache evita que ela seja
# reexecutada a cada interação no dashboard, tornando a experiência mais fluida.

def encontrar_k_otimo(_df_padronizado, max_k=10):
    """
    Calcula a inércia (WCSS) e o coeficiente de silhueta para um range de K.
    O _ antes do nome do DataFrame é uma convenção para indicar ao Streamlit
    que não monitore mudanças no conteúdo do DataFrame para o cache, apenas
    a sua identidade, o que melhora a performance.
    """
    inercias = []
    scores_silhueta = []
    range_k = range(2, max_k + 1)
    
    for k in range_k:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(_df_padronizado)
        inercias.append(kmeans.inertia_)
        # Calcula o score de silhueta para os labels gerados
        labels = kmeans.labels_
        scores_silhueta.append(silhouette_score(_df_padronizado, labels))
        
    print("Cálculo de inércia e scores de silhueta concluído.")
    return {'range_k': list(range_k), 'inercias': inercias, 'scores_silhueta': scores_silhueta}

def aplicar_kmeans(df_padronizado, n_clusters=4):
    """
    Aplica o algoritmo K-Means.
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_padronizado)
    print(f"K-Means aplicado com {n_clusters} clusters.")
    return labels

def aplicar_cluster_hierarquico(df_padronizado, n_clusters=4):
    """
    Aplica o algoritmo de Clusterização Hierárquica Aglomerativa.
    """
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg_clustering.fit_predict(df_padronizado)
    print(f"Clusterização Hierárquica aplicada com {n_clusters} clusters.")
    return labels, agg_clustering

def aplicar_dbscan(df_padronizado, eps=0.5, min_samples=5):
    """
    Aplica o algoritmo DBSCAN.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(df_padronizado)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"DBSCAN aplicado com eps={eps} e min_samples={min_samples}.")
    print(f"Número de clusters encontrados: {n_clusters}")
    print(f"Número de pontos de ruído: {n_noise}")
    return labels

