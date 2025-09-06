# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import preprocessing # Importação necessária para a normalização do radar

def plotar_matriz_correlacao(df_numerico):
    """
    Cria e retorna um gráfico de mapa de calor da matriz de correlação.
    Ajustado para um número maior de variáveis.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    # Aumentando o tamanho da figura para melhor visualização
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df_numerico.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
    ax.set_title('Matriz de Correlação das Variáveis Numéricas', fontsize=16)
    return fig

def plotar_distribuicoes(df_numerico):
    """
    Cria e retorna uma figura com histogramas para cada variável numérica.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    num_plots = len(df_numerico.columns)
    
    # Layout dinâmico com 2 colunas
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(df_numerico.columns):
        sns.histplot(df_numerico[col], kde=True, ax=axes[i], bins=50, color='royalblue')
        axes[i].set_title(f'Distribuição de: {col}', fontsize=12)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    # Oculta eixos não utilizados se o número de plots for ímpar
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(pad=3.0)
    return fig

def plotar_metodo_cotovelo(resultados_k):
    """
    Plota o gráfico do Método do Cotovelo.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(resultados_k['range_k'], resultados_k['inercias'], 'bo-')
    ax.set_xlabel('Número de Clusters (K)')
    ax.set_ylabel('Inércia (WCSS)')
    ax.set_title('Método do Cotovelo para K Ótimo')
    return fig

def plotar_score_silhueta(resultados_k):
    """
    Plota o gráfico do Coeficiente de Silhueta.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(resultados_k['range_k'], resultados_k['scores_silhueta'], 'ro-')
    ax.set_xlabel('Número de Clusters (K)')
    ax.set_ylabel('Coeficiente Médio de Silhueta')
    ax.set_title('Análise de Silhueta para K Ótimo')
    return fig

def plotar_cluster_pca_individual(df_padronizado, labels, nome_modelo):
    """
    Plota a visualização de um único modelo de cluster usando PCA.
    """
    pca = PCA(n_components=2)
    componentes_principais = pca.fit_transform(df_padronizado)
    df_pca = pd.DataFrame(data=componentes_principais, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = labels
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # A paleta 'tab10' é uma boa escolha para até 10 clusters distintos
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, palette='tab10', alpha=0.6, s=30, ax=ax)
    ax.set_title(f'Clusters do Modelo: {nome_modelo}')
    ax.grid(True)
    ax.legend(title='Cluster')
    return fig

def plotar_radar_individual(perfil_clusters, cluster_id):
    """
    Plota um gráfico de radar para o perfil de um único cluster.
    """
    # A análise de perfil já vem apenas com as features numéricas originais
    perfil_para_plot = perfil_clusters.drop(columns=['n_clientes'])
    perfil_normalizado = preprocessing.normalizar_para_radar(perfil_para_plot)
    
    labels = perfil_normalizado.columns
    stats = perfil_normalizado.loc[cluster_id].values
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats = np.concatenate((stats,[stats[0]]))
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, stats, color='darkviolet', linewidth=2)
    ax.fill(angles, stats, color='darkviolet', alpha=0.25)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_title(f'Perfil do Cluster {cluster_id}', size=14, pad=20)
    
    return fig