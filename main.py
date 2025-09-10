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
        print("Arquivo '{}' encontrado. Carregando dados existentes...".format(nome_arquivo))
        df_clientes = pd.read_excel(nome_arquivo)
    else:
        print("Arquivo '{}' não encontrado. Gerando e salvando nova base de dados...".format(nome_arquivo))
        df_clientes = data_generator.gerar_dados_sinteticos(n_clientes=30000, seed=42)
        df_clientes.to_excel(nome_arquivo, index=False)
        print("Base de dados sintética salva com sucesso.")
    
    # Etapa 2: Pré-processamento
    print("\n--- Iniciando o pré-processamento dos dados ---")
    df_numerico_original, df_para_modelagem = preprocessing.selecionar_e_transformar_features(df_clientes)
    df_padronizado = preprocessing.padronizar_dados(df_para_modelagem)
    
    # --- NOVO: Visualizações de dados brutos ---
    print("\n--- Gerando matriz de correlação ---")
    visualization.plotar_matriz_correlacao(df_numerico_original)
    plt.show()
    
    print("\n--- Gerando distribuições das variáveis numéricas ---")
    colunas_demograficas = ['idade', 'numero_dependentes']
    colunas_financeiras = ['renda_mensal', 'score_credito', 'historico_pagamento_recente', 
                       'tempo_de_debito_meses', 'valor_divida']

    visualization.plotar_distribuicoes_separadas(df_numerico_original, colunas_demograficas, colunas_financeiras)

    print("\n--- Gerando distribuições das variáveis categóricas ---")

    colunas_categoricas = ['sexo', 'estado_civil', 'nivel_educacional', 'tipo_emprego']
    visualization.plotar_categoricas(df_clientes, colunas_categoricas, filename_prefix="distribuicoes_categoricas")

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
    print("\nNúmero de clusters escolhido com base na análise: {}".format(K_OTIMO))
    
    # Etapa 4: Aplicação dos Modelos
    print("\n--- Aplicando os modelos de clusterização ---")
    labels_kmeans = clustering_models.aplicar_kmeans(df_padronizado, n_clusters=K_OTIMO)

    print("Aplicando Clusterização Hierárquica em uma amostra...")
    if len(df_padronizado) > 10000:
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
    labels_dict_full = {
        'K-Means': labels_kmeans,
        'DBSCAN': labels_dbscan
    }
    df_avaliacao_full = evaluation.avaliar_modelos(df_padronizado, labels_dict_full)
    labels_dict_sample = {'Hierárquico': labels_hierarquico}
    df_avaliacao_sample = evaluation.avaliar_modelos(df_amostra, labels_dict_sample)
    df_avaliacao_final = pd.concat([df_avaliacao_full, df_avaliacao_sample])
    print("\nTabela de Avaliação Comparativa dos Modelos:")
    print(df_avaliacao_final.sort_values(by='Coeficiente de Silhueta', ascending=False).to_string())

    # Etapa 6: Visualização e Análise de Perfis
    print("\n--- Gerando visualizações dos clusters via PCA ---")
    for nome_modelo, labels in labels_dict_full.items():
        if len(set(labels)) > 1:
            visualization.plotar_cluster_pca_individual(df_padronizado, labels, nome_modelo)
            plt.show()

    if 'Hierárquico' in labels_dict_sample:
        labels_hierarquico = labels_dict_sample['Hierárquico']
        if len(set(labels_hierarquico)) > 1:
            visualization.plotar_cluster_pca_individual(df_amostra, labels_hierarquico, "Hierárquico")
            plt.show()

    print("\n--- Análise de Perfil dos Clusters (K-Means) ---")
    perfil_kmeans = evaluation.analisar_perfis_clusters(df_numerico_original, labels_kmeans, 'K-Means')
    print(perfil_kmeans.to_string())
    perfil_cat_kmeans = evaluation.analisar_perfis_categoricos(df_clientes, labels_kmeans, 'K-Means')
    print(perfil_cat_kmeans.to_string())

    print("\n--- Análise de Perfil dos Clusters (Hierárquico na Amostra) ---")
    df_clientes_amostra = df_clientes.loc[df_amostra.index]
    df_numerico_original_amostra = df_numerico_original.loc[df_amostra.index]
    perfil_hierarquico_num = evaluation.analisar_perfis_clusters(df_numerico_original_amostra, labels_hierarquico, 'Hierárquico')
    print(perfil_hierarquico_num.to_string())
    perfil_hierarquico_cat = evaluation.analisar_perfis_categoricos(df_clientes_amostra, labels_hierarquico, 'Hierárquico')
    print(perfil_hierarquico_cat.to_string())

if __name__ == '__main__':
    main()