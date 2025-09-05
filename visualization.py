import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import pandas as pd

def plotar_metodo_cotovelo(resultados_k):
    """
    Plota o gráfico do Método do Cotovelo (Elbow Method).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(resultados_k['range_k'], resultados_k['inercias'], marker='o', linestyle='--')
    plt.title('Método do Cotovelo (Elbow Method)')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inércia (WCSS)')
    plt.grid(True)
    plt.savefig('metodo_cotovelo.png')
    plt.show()

def plotar_score_silhueta(resultados_k):
    """
    Plota o gráfico do Coeficiente de Silhueta para diferentes valores de K.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(resultados_k['range_k'], resultados_k['scores_silhueta'], marker='o', linestyle='--')
    plt.title('Coeficiente de Silhueta para Diferentes K')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig('score_silhueta.png')
    plt.show()

def plotar_dendrograma(modelo_hierarquico, df_padronizado):
    """
    Plota o dendrograma para o modelo de clusterização hierárquica.
    """
    # Recalcula a ligação (linkage) para plotagem
    linked = linkage(df_padronizado, method='ward')
    
    plt.figure(figsize=(15, 7))
    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True,
               truncate_mode='level',
               p=5)
    plt.title('Dendrograma da Clusterização Hierárquica')
    plt.xlabel("Índice do Ponto de Dados (ou tamanho do cluster)")
    plt.ylabel("Distância")
    plt.savefig('dendrograma.png')
    plt.show()

def plotar_clusters_pca(df_padronizado, labels_dict):
    """
    Aplica PCA e plota os clusters em 2D.
    """
    pca = PCA(n_components=2)
    componentes_principais = pca.fit_transform(df_padronizado)
    df_pca = pd.DataFrame(data=componentes_principais, columns=['Componente Principal 1', 'Componente Principal 2'])
    
    for nome_modelo, labels in labels_dict.items():
        df_pca[f'cluster_{nome_modelo}'] = labels
        
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='Componente Principal 1', y='Componente Principal 2', hue=f'cluster_{nome_modelo}', 
                        palette=sns.color_palette("hsv", len(np.unique(labels))),
                        data=df_pca, legend='full', alpha=0.7)
        plt.title(f'Visualização 2D dos Clusters ({nome_modelo}) via PCA')
        plt.savefig(f'clusters_pca_{nome_modelo}.png')
        plt.show()
