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
    com a base de dados de 20.000 registros.
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
        df_clientes = data_generator.gerar_dados_sinteticos(n_clientes=30000, seed=42)
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

    # Hierárquico (APENAS em uma amostra)
    print("Aplicando Clusterização Hierárquica em uma amostra...")
    if len(df_padronizado) > 10000:
        # Pega uma amostra de 10.000 pontos para não estourar a memória
        df_amostra = df_padronizado.sample(n=10000, random_state=42)
    else:
        df_amostra = df_padronizado

    labels_hierarquico, _ = clustering_models.aplicar_cluster_hierarquico(df_amostra, n_clusters=K_OTIMO)
    labels_dbscan = clustering_models.aplicar_dbscan(df_padronizado, eps=2.5, min_samples=20)
    
    labels_dict = {
        'K-Means': labels_kmeans,
        'Hierárquico': labels_hierarquico,
        'DBSCAN': labels_dbscan
    }
    
    # Etapa 5: Avaliação
    print("\n--- Avaliando os modelos ---")

    # Dicionário para modelos que usaram o DATASET COMPLETO
    labels_dict_full = {
        'K-Means': labels_kmeans,
        'DBSCAN': labels_dbscan
    }
    print("\nAvaliando modelos no dataset completo (20.000 amostras):")
    df_avaliacao_full = evaluation.avaliar_modelos(df_padronizado, labels_dict_full)

    # Dicionário para modelos que usaram a AMOSTRA
    labels_dict_sample = {
        'Hierárquico': labels_hierarquico,
    }
    print("\nAvaliando modelos na amostra (10.000 amostras):")
    df_avaliacao_sample = evaluation.avaliar_modelos(df_amostra, labels_dict_sample)

    # Juntando os resultados para uma única tabela de comparação
    df_avaliacao_final = pd.concat([df_avaliacao_full, df_avaliacao_sample])

    print("\nTabela de Avaliação Comparativa dos Modelos:")
    # O .sort_values() é opcional, mas ajuda a manter a tabela organizada
    print(df_avaliacao_final.sort_values(by='Coeficiente de Silhueta', ascending=False).to_string())

    # Etapa 6: Visualização e Análise de Perfis
    print("\n--- Gerando visualizações dos clusters via PCA ---")

    # Visualização para modelos do dataset completo
    for nome_modelo, labels in labels_dict_full.items():
        if len(set(labels)) > 1:
            print("Salvando clusters para o modelo (dataset completo): {}".format(nome_modelo))
            # Passamos o df_padronizado aqui
            visualization.plotar_cluster_pca_individual(df_padronizado, labels, nome_modelo)

    # Visualização para o modelo da amostra
    if len(set(labels_hierarquico)) > 1:
        print("Salvando clusters para o modelo (amostra): Hierárquico")
        # ATENÇÃO: Passamos o df_amostra aqui
        visualization.plotar_cluster_pca_individual(df_amostra, labels_hierarquico, "Hierárquico")
    
    # Etapa 6: Visualização e Análise de Perfis
    print("\n--- Gerando visualizações dos clusters via PCA ---")

    # Visualização para modelos do dataset completo (K-Means, DBSCAN)
    for nome_modelo, labels in labels_dict_full.items():
        if len(set(labels)) > 1:
            print("Salvando clusters para o modelo (dataset completo): {}".format(nome_modelo))
            # Correto: Usa df_padronizado (20k) com labels de 20k
            visualization.plotar_cluster_pca_individual(df_padronizado, labels, nome_modelo)

    # Visualização para o modelo da amostra (Hierárquico)
    # Checamos se o dicionário existe para evitar erros
    if 'Hierárquico' in labels_dict_sample:
        labels_hierarquico = labels_dict_sample['Hierárquico']
        if len(set(labels_hierarquico)) > 1:
            print("Salvando clusters para o modelo (amostra): Hierárquico")
            # A CORREÇÃO PRINCIPAL ESTÁ AQUI:
            # Usamos df_amostra (10k) com os labels do hierárquico (10k)
            visualization.plotar_cluster_pca_individual(df_amostra, labels_hierarquico, "Hierárquico")

    print("\n--- Análise de Perfil dos Clusters (K-Means) ---")
    perfil_kmeans = evaluation.analisar_perfis_clusters(df_numerico_original, labels_kmeans, 'K-Means')
    print("\nPerfil Numérico Médio dos Clusters (K-Means):")
    print(perfil_kmeans.to_string())

    # --- NOVO: Análise de Perfil Categórico (K-Means) ---
    perfil_cat_kmeans = evaluation.analisar_perfis_categoricos(df_clientes, labels_kmeans, 'K-Means')
    print("\nPerfil Categórico (Moda) dos Clusters (K-Means):")
    print(perfil_cat_kmeans.to_string())
    # --- FIM DA ADIÇÃO ---


    # --- NOVO: Análise de Perfil para o modelo Hierárquico (amostra) ---
    print("\n--- Análise de Perfil dos Clusters (Hierárquico na Amostra) ---")
    
    # Precisamos pegar as linhas originais correspondentes à amostra
    df_clientes_amostra = df_clientes.loc[df_amostra.index]
    df_numerico_original_amostra = df_numerico_original.loc[df_amostra.index]
    
    perfil_hierarquico_num = evaluation.analisar_perfis_clusters(df_numerico_original_amostra, labels_hierarquico, 'Hierárquico')
    print("\nPerfil Numérico Médio dos Clusters (Hierárquico na Amostra):")
    print(perfil_hierarquico_num.to_string())
    
    perfil_hierarquico_cat = evaluation.analisar_perfis_categoricos(df_clientes_amostra, labels_hierarquico, 'Hierárquico')
    print("\nPerfil Categórico (Moda) dos Clusters (Hierárquico na Amostra):")
    print(perfil_hierarquico_cat.to_string())
    # --- FIM DA ADIÇÃO ---


if __name__ == '__main__':
    main()
