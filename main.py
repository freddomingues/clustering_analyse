# -*- coding: utf-8 -*-
import pandas as pd
import os
import matplotlib.pyplot as plt

# Importando os módulos do projeto
import data_generator
import preprocessing
import clustering_models
import evaluation
import visualization

def main():
    """
    Função principal para executar o pipeline completo de clusterização
    com a base de dados de 100.000 registros.
    """
    nome_arquivo = 'base_sintetica_dividas.xlsx'

    # Etapa 1: Geração ou Carregamento dos Dados
    if os.path.exists(nome_arquivo):
        # >>> CORREÇÃO: Trocando f-string por .format() <<<
        print("Arquivo '{}' encontrado. Carregando dados existentes...".format(nome_arquivo))
        df_clientes = pd.read_excel(nome_arquivo)
    else:
        # >>> CORREÇÃO: Trocando f-string por .format() <<<
        print("Arquivo '{}' não encontrado. Gerando e salvando nova base de dados...".format(nome_arquivo))
        df_clientes = data_generator.gerar_dados_sinteticos(n_clientes=100000, seed=42)
        df_clientes.to_excel(nome_arquivo, index=False)
        print("Base de dados sintética salva com sucesso.")
    
    # Etapa 2: Pré-processamento
    print("\n--- Iniciando o pré-processamento dos dados ---")
    df_numerico_original, df_para_modelagem = preprocessing.selecionar_e_transformar_features(df_clientes)
    df_padronizado = preprocessing.padronizar_dados(df_para_modelagem)
    
    # Etapa 3: Determinação do K ótimo para K-Means
    print("\n--- Iniciando a determinação do K ótimo para K-Means ---")
    resultados_k = clustering_models.encontrar_k_otimo(df_padronizado, max_k=10)
    
    print("Exibindo gráfico do Método do Cotovelo...")
    visualization.plotar_metodo_cotovelo(resultados_k)
    plt.show()
    
    print("Exibindo gráfico da Análise de Silhueta...")
    visualization.plotar_score_silhueta(resultados_k)
    plt.show()
    
    K_OTIMO = 4
    # >>> CORREÇÃO: Trocando f-string por .format() <<<
    print("\nNúmero de clusters escolhido com base na análise: {}".format(K_OTIMO))
    
    # Etapa 4: Aplicação dos Modelos
    print("\n--- Aplicando os modelos de clusterização ---")
    labels_kmeans = clustering_models.aplicar_kmeans(df_padronizado, n_clusters=K_OTIMO)
    labels_hierarquico, _ = clustering_models.aplicar_cluster_hierarquico(df_padronizado, n_clusters=K_OTIMO)
    labels_dbscan = clustering_models.aplicar_dbscan(df_padronizado, eps=2.5, min_samples=20)
    
    labels_dict = {
        'K-Means': labels_kmeans,
        'Hierárquico': labels_hierarquico,
        'DBSCAN': labels_dbscan
    }
    
    # Etapa 5: Avaliação
    print("\n--- Avaliando os modelos ---")
    df_avaliacao = evaluation.avaliar_modelos(df_padronizado, labels_dict)
    print("\nTabela de Avaliação Comparativa dos Modelos:")
    print(df_avaliacao.to_string())
    
    # Etapa 6: Visualização e Análise de Perfis
    print("\n--- Gerando visualizações dos clusters via PCA ---")
    for nome_modelo, labels in labels_dict.items():
        if len(set(labels)) > 1:
            # >>> CORREÇÃO: Trocando f-string por .format() <<<
            print("Exibindo clusters para o modelo: {}".format(nome_modelo))
            visualization.plotar_cluster_pca_individual(df_padronizado, labels, nome_modelo)
            plt.show()

    print("\n--- Análise de Perfil dos Clusters (K-Means) ---")
    perfil_kmeans = evaluation.analisar_perfis_clusters(df_numerico_original, labels_kmeans, 'K-Means')
    print("\nPerfil Médio dos Clusters Gerados pelo K-Means:")
    print(perfil_kmeans.to_string())

if __name__ == '__main__':
    main()

