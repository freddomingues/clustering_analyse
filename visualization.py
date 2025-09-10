# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import os
import preprocessing  # Importação necessária para a normalização do radar

# Cria o diretório 'images' se não existir
os.makedirs("images", exist_ok=True)

def plotar_matriz_correlacao(df_numerico, filename="matriz_correlacao.png"):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df_numerico.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
    ax.set_title('Matriz de Correlação das Variáveis Numéricas', fontsize=16)
    fig.tight_layout()
    fig.savefig(f"images/{filename}")
    plt.close(fig)  # Fecha a figura para liberar memória

def plotar_distribuicoes(df_numerico, filename_prefix="distribuicao"):
    plt.style.use('seaborn-v0_8-whitegrid')
    num_plots = len(df_numerico.columns)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(df_numerico.columns):
        sns.histplot(df_numerico[col], kde=True, ax=axes[i], bins=50, color='royalblue')
        axes[i].set_title(f'Distribuição de: {col}', fontsize=12)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(pad=3.0)
    fig.savefig(f"images/{filename_prefix}.png")
    plt.close(fig)

def plotar_metodo_cotovelo(resultados_k, filename="metodo_cotovelo.png"):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(resultados_k['range_k'], resultados_k['inercias'], 'bo-')
    ax.set_xlabel('Número de Clusters (K)')
    ax.set_ylabel('Inércia (WCSS)')
    ax.set_title('Método do Cotovelo para K Ótimo')
    fig.tight_layout()
    fig.savefig(f"images/{filename}")
    plt.close(fig)

def plotar_score_silhueta(resultados_k, filename="score_silhueta.png"):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(resultados_k['range_k'], resultados_k['scores_silhueta'], 'ro-')
    ax.set_xlabel('Número de Clusters (K)')
    ax.set_ylabel('Coeficiente Médio de Silhueta')
    ax.set_title('Análise de Silhueta para K Ótimo')
    fig.tight_layout()
    fig.savefig(f"images/{filename}")
    plt.close(fig)

def plotar_cluster_pca_individual(df_padronizado, labels, nome_modelo, filename=None):
    if filename is None:
        filename = f"clusters_pca_{nome_modelo}.png"
    
    pca = PCA(n_components=2)
    componentes_principais = pca.fit_transform(df_padronizado)
    df_pca = pd.DataFrame(data=componentes_principais, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = labels
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, palette='tab10', alpha=0.6, s=30, ax=ax)
    ax.set_title(f'Clusters do Modelo: {nome_modelo}')
    ax.grid(True)
    ax.legend(title='Cluster')
    fig.tight_layout()
    fig.savefig(f"images/{filename}")
    plt.close(fig)

def plotar_radar_individual(perfil_clusters, cluster_id, filename=None):
    if filename is None:
        filename = f"radar_cluster_{cluster_id}.png"
    
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
    
    fig.tight_layout()
    fig.savefig(f"images/{filename}")
    plt.close(fig)