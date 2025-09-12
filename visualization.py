# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import os
import preprocessing  # Importação necessária para a normalização do radar
from math import pi

# Cria o diretório 'images' se não existir
os.makedirs("images", exist_ok=True)

def salvar_tabela_descritiva(df_numerico, filename="tabela_descritiva.csv"):
    """
    Salva a tabela de estatísticas descritivas das variáveis numéricas.
    """
    tabela = df_numerico.describe().transpose()
    tabela.to_csv(f"images/{filename}")

def plotar_matriz_correlacao(df_numerico, filename="matriz_correlacao.png"):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df_numerico.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, vmin=-1, vmax=1, cbar_kws={'label': 'Correlação'})
    ax.set_title('Matriz de Correlação das Variáveis Numéricas', fontsize=16)
    fig.tight_layout()
    fig.savefig(f"images/{filename}")
    plt.close(fig)

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
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequência')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(pad=3.0)
    fig.savefig(f"images/{filename_prefix}.png")
    plt.close(fig)

def plotar_metodo_cotovelo(resultados_k, filename="metodo_cotovelo.png"):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(resultados_k['range_k'], resultados_k['inercias'], 'bo-', label='WCSS')
    ax.set_xlabel('Número de Clusters (K)')
    ax.set_ylabel('Inércia (WCSS)')
    ax.set_title('Método do Cotovelo para Determinação do K Ótimo', fontsize=14)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"images/{filename}")
    plt.close(fig)

def plotar_score_silhueta(resultados_k, filename="score_silhueta.png"):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(resultados_k['range_k'], resultados_k['scores_silhueta'], 'ro-', label='Coeficiente de Silhueta')
    ax.set_xlabel('Número de Clusters (K)')
    ax.set_ylabel('Coeficiente Médio de Silhueta')
    ax.set_title('Análise de Silhueta para Determinação do K Ótimo', fontsize=14)
    ax.grid(True)
    ax.legend()
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
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, palette='tab10', alpha=0.7, s=50, ax=ax)
    ax.set_title(f'Clusters do Modelo: {nome_modelo}', fontsize=14)
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
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
    ax.plot(angles, stats, color='darkviolet', linewidth=2, label=f'Cluster {cluster_id}')
    ax.fill(angles, stats, color='darkviolet', alpha=0.25)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_title(f'Perfil do Cluster {cluster_id}', size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    fig.tight_layout()
    fig.savefig(f"images/{filename}")
    plt.close(fig)

def plotar_distribuicoes_separadas(df_numerico, colunas_demograficas, colunas_financeiras):
    """
    Plota distribuições das variáveis numéricas separando demográficas e financeiras.
    Salva duas imagens: uma para demográficas e outra para financeiras.
    """
    def plotar(df_subset, filename):
        plt.style.use('seaborn-v0_8-whitegrid')
        num_plots = len(df_subset.columns)
        num_cols = 2
        num_rows = (num_plots + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 4))
        axes = axes.flatten()
    
        for i, col in enumerate(df_subset.columns):
            sns.histplot(df_subset[col], kde=True, ax=axes[i], bins=50, color='royalblue')
            axes[i].set_title(f'Distribuição de: {col}', fontsize=12)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequência')
    
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
    
        fig.tight_layout(pad=3.0)
        fig.savefig(f"images/{filename}")
        plt.close(fig)
    
    # Plotando variáveis demográficas
    if colunas_demograficas:
        plotar(df_numerico[colunas_demograficas], "distribuicoes_demograficas.png")
    
    # Plotando variáveis financeiras
    if colunas_financeiras:
        plotar(df_numerico[colunas_financeiras], "distribuicoes_financeiras.png")

def plotar_categoricas(df, colunas_categoricas, filename_prefix="categoricas"):
    """
    Plota contagens das variáveis categóricas em gráficos de barra.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    num_plots = len(colunas_categoricas)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(colunas_categoricas):
        sns.countplot(x=col, data=df, hue=col, ax=axes[i], palette='pastel', legend=False)
        axes[i].set_title(f'Contagem de: {col}', fontsize=12)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequência')
        for p in axes[i].patches:
            axes[i].annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=10)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(pad=3.0)
    fig.savefig(f"images/{filename_prefix}.png")
    plt.close(fig)

def plotar_cotovelo_e_silhueta_juntos(resultados_k, filename="cotovelo_silhueta.png"):
    """
    Plota lado a lado o gráfico do método do cotovelo e o gráfico do coeficiente de silhueta.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Gráfico do Cotovelo
    axes[0].plot(resultados_k['range_k'], resultados_k['inercias'], 'bo-', label='WCSS')
    axes[0].set_xlabel('Número de Clusters (K)')
    axes[0].set_ylabel('Inércia (WCSS)')
    axes[0].set_title('Método do Cotovelo', fontsize=14)
    axes[0].grid(True)
    axes[0].legend()

    # Gráfico da Silhueta
    axes[1].plot(resultados_k['range_k'], resultados_k['scores_silhueta'], 'ro-', label='Coeficiente de Silhueta')
    axes[1].set_xlabel('Número de Clusters (K)')
    axes[1].set_ylabel('Coeficiente Médio de Silhueta')
    axes[1].set_title('Análise de Silhueta', fontsize=14)
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(f"images/{filename}")
    plt.close(fig)

def normalizar_por_variavel(df):
    """
    Normaliza cada coluna (variável) de um DataFrame para a escala [0, 1],
    considerando apenas os valores médios por cluster.
    """
    return (df - df.min()) / (df.max() - df.min())

def plotar_radar_clusters(df_clusters, features, n_clusters, output_dir="images", filename="radar_clusters.png"):
    """
    Plota os gráficos de radar dos clusters em uma única figura com quadrantes.
    Normaliza cada variável para [0,1] considerando apenas os valores médios dos clusters.
    """
    import numpy as np
    import matplotlib.cm as cm

    os.makedirs(output_dir, exist_ok=True)

    # Calcula médias por cluster
    cluster_means = df_clusters.groupby('cluster')[features].mean()

    # Normaliza cada variável (coluna) de forma independente
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

    num_vars = len(features)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Cores diferentes para cada cluster
    colors = cm.get_cmap("tab10", n_clusters)

    # Subplots em 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    for cluster_id in range(n_clusters):
        valores = cluster_means_norm.loc[cluster_id].values.tolist()
        valores += valores[:1]

        ax = axes[cluster_id]
        ax.plot(angles, valores, linewidth=2, label=f'Cluster {cluster_id}', color=colors(cluster_id))
        ax.fill(angles, valores, alpha=0.25, color=colors(cluster_id))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, size=9)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0], size=8)
        ax.set_ylim(0, 1)
        ax.set_title(f'Cluster {cluster_id}', size=12, pad=15)

    for j in range(cluster_id + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Perfis Normalizados dos Clusters (comparação por variável)", size=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)