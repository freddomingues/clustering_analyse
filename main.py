# -*- coding: utf-8 -*-
import pandas as pd
import os  # Módulo necessário para interagir com o sistema operacional (verificar arquivos)

import data_generator
import preprocessing
import clustering_models
import evaluation
import visualization

def main():
    """
    Função principal para executar o pipeline completo de clusterização.
    """
    nome_arquivo = 'base_sintetica_dividas.xlsx'

    # >>> LÓGICA ATUALIZADA: VERIFICAR SE O ARQUIVO JÁ EXISTE <<<
    if os.path.exists(nome_arquivo):
        # Se o arquivo já existe, apenas o carrega para o DataFrame.
        print("Arquivo '{}' encontrado. Carregando dados existentes...".format(nome_arquivo))
        df_clientes = pd.read_excel(nome_arquivo)
    else:
        # Se o arquivo não existe, gera os dados e os salva.
        print("Arquivo '{}' não encontrado. Gerando e salvando nova base de dados...".format(nome_arquivo))
        df_clientes = data_generator.gerar_dados_sinteticos(n_clientes=20000, seed=42)
        df_clientes.to_excel(nome_arquivo, index=False)
        print("Base de dados sintética salva com sucesso.")
    
    # Etapa 2: Pré-processamento
    df_numerico = preprocessing.selecionar_features(df_clientes)
    df_padronizado = preprocessing.padronizar_dados(df_numerico)
    
    # Etapa 3: Determinação do K ótimo para K-Means
    print("\n--- Iniciando a determinação do K ótimo para K-Means ---")
    resultados_k = clustering_models.encontrar_k_otimo(df_padronizado, max_k=10)
    visualization.plotar_metodo_cotovelo(resultados_k)
    visualization.plotar_score_silhueta(resultados_k)
    
    K_OTIMO = 4
    print("\nNúmero de clusters escolhido: {}".format(K_OTIMO))
    
    # Etapa 4: Aplicação dos Modelos
    print("\n--- Aplicando os modelos de clusterização ---")
    labels_kmeans = clustering_models.aplicar_kmeans(df_padronizado, n_clusters=K_OTIMO)
    labels_hierarquico, modelo_hierarquico = clustering_models.aplicar_cluster_hierarquico(df_padronizado, n_clusters=K_OTIMO)
    labels_dbscan = clustering_models.aplicar_dbscan(df_padronizado, eps=1.5, min_samples=10)
    
    labels_dict = {
        'KMeans': labels_kmeans,
        'Hierarquico': labels_hierarquico,
        'DBSCAN': labels_dbscan
    }
    
    # Etapa 5: Avaliação
    print("\n--- Avaliando os modelos ---")
    df_avaliacao = evaluation.avaliar_modelos(df_padronizado, labels_dict)
    print("\nTabela de Avaliação Comparativa dos Modelos:")
    print(df_avaliacao.to_string())
    
    # Etapa 6: Visualização e Análise de Perfis
    print("\n--- Gerando visualizações e perfis ---")
    visualization.plotar_clusters_pca(df_padronizado, labels_dict)
    visualization.plotar_dendrograma(modelo_hierarquico, df_padronizado)
    
    print("\n--- Análise de Perfil dos Clusters (K-Means) ---")
    perfil_kmeans = evaluation.analisar_perfis_clusters(df_numerico, labels_kmeans, 'KMeans')
    print("\nPerfil Médio dos Clusters Gerados pelo K-Means:")
    print(perfil_kmeans.to_string())

if __name__ == '__main__':
    main()