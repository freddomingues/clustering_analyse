# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import os
import time

# Importando nossos módulos
import data_generator
import preprocessing
import clustering_models
import evaluation
import visualization

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Dashboard de Segmentação de Clientes",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Funções com Cache para Performance ---
# O cache do Streamlit armazena o resultado da função.
# Assim, os dados não são gerados/carregados toda vez que interagimos com o app.

@st.cache_data
def carregar_ou_gerar_dados(n_clientes, seed):
    """Verifica se a base existe, senão, a gera."""
    nome_arquivo = 'base_sintetica_dividas.xlsx'
    if not os.path.exists(nome_arquivo):
        with st.spinner('Base de dados não encontrada. Gerando dados sintéticos... Isso pode levar um momento.'):
            df = data_generator.gerar_dados_sinteticos(n_clientes=n_clientes, seed=seed)
            df.to_excel(nome_arquivo, index=False)
        return df
    return pd.read_excel(nome_arquivo)

@st.cache_data
def processar_dados(df):
    """Função para encapsular todo o pré-processamento."""
    df_numerico = preprocessing.selecionar_features(df)
    df_padronizado = preprocessing.padronizar_dados(df_numerico)
    return df_numerico, df_padronizado

# --- Título e Descrição ---
st.title('👥 Dashboard Interativo para Segmentação de Clientes')
st.markdown("""
Esta ferramenta permite explorar a segmentação de clientes inadimplentes utilizando diferentes algoritmos de clusterização. 
Use a barra lateral para configurar os parâmetros e navegue pelas abas para visualizar os resultados.
""")

# --- Barra Lateral de Controles ---
with st.sidebar:
    st.header('⚙️ Parâmetros de Análise')

    # Parâmetros para geração de dados (só serão usados se o arquivo não existir)
    n_clientes = st.slider('Número de Clientes (para 1ª geração)', 5000, 50000, 20000, 1000)
    seed = st.number_input('Semente Aleatória (Seed)', value=42, step=1)

    st.markdown("---")

    # Parâmetros para os modelos de clusterização
    st.header('🤖 Parâmetros dos Modelos')
    k_otimo = st.slider('Número de Clusters (K) para K-Means e Hierárquico', min_value=2, max_value=10, value=4)
    dbscan_eps = st.slider('Raio da Vizinhança (eps) para DBSCAN', min_value=0.1, max_value=3.0, value=1.5, step=0.1)
    dbscan_min_samples = st.slider('Nº Mínimo de Amostras (min_samples) para DBSCAN', min_value=5, max_value=50, value=10, step=1)
    
    st.info("O K ótimo pode ser analisado na aba 'Definição do Número de Clusters (K)'.")


# --- Carregamento e Processamento dos Dados ---
with st.spinner('Carregando e processando os dados...'):
    df_clientes = carregar_ou_gerar_dados(n_clientes, seed)
    df_numerico, df_padronizado = processar_dados(df_clientes)

# --- Corpo Principal com Abas ---
st.header("Análise de Clusterização")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Visão Geral dos Dados",
    "📈 Definição do Número de Clusters (K)",
    "🤖 Resultados dos Modelos",
    "🔍 Análise de Perfil dos Clusters (K-Means)",
    "ℹ️ Sobre o Projeto"
])

with tab1:
    st.subheader("Amostra da Base de Dados")
    st.dataframe(df_clientes.head(10))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Estatísticas Descritivas")
        st.dataframe(df_numerico.describe())
    with col2:
        st.subheader("Matriz de Correlação")
        fig_corr = visualization.plotar_matriz_correlacao(df_numerico)
        st.pyplot(fig_corr)

with tab2:
    st.subheader("Análise para Determinação do K Ótimo (K-Means)")
    with st.spinner("Calculando o K ótimo... Isso pode levar alguns segundos."):
        resultados_k = clustering_models.encontrar_k_otimo(df_padronizado, max_k=10)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Método do Cotovelo (Elbow Method)")
        fig_cotovelo = visualization.plotar_metodo_cotovelo(resultados_k)
        st.pyplot(fig_cotovelo)
        st.info("O 'cotovelo' (ponto de inflexão) sugere um bom número de clusters. Neste caso, parece estar em K=4.")
    with col2:
        st.subheader("Coeficiente de Silhueta")
        fig_silhueta = visualization.plotar_score_silhueta(resultados_k)
        st.pyplot(fig_silhueta)
        st.info("O pico do gráfico indica o melhor K em termos de coesão e separação dos clusters. K=4 também se destaca aqui.")

with tab3:
    st.subheader("Comparativo dos Modelos de Clusterização")
    
    with st.spinner("Treinando e avaliando os modelos..."):
        # Aplicação dos modelos com os parâmetros da barra lateral
        labels_kmeans = clustering_models.aplicar_kmeans(df_padronizado, n_clusters=k_otimo)
        labels_hierarquico, modelo_hierarquico = clustering_models.aplicar_cluster_hierarquico(df_padronizado, n_clusters=k_otimo)
        labels_dbscan = clustering_models.aplicar_dbscan(df_padronizado, eps=dbscan_eps, min_samples=dbscan_min_samples)

        labels_dict = {
            'KMeans': labels_kmeans,
            'Hierarquico': labels_hierarquico,
            'DBSCAN': labels_dbscan
        }
    
    st.subheader("Visualização dos Clusters (via PCA)")
    fig_pca = visualization.plotar_clusters_pca(df_padronizado, labels_dict)
    st.pyplot(fig_pca)

    st.subheader("Métricas de Avaliação")
    df_avaliacao = evaluation.avaliar_modelos(df_padronizado, labels_dict)
    st.dataframe(df_avaliacao.style.highlight_max(subset=['Coeficiente de Silhueta'], color='lightgreen').highlight_min(subset=['Índice de Davies-Bouldin'], color='lightgreen'))
    st.markdown("""
    - **Coeficiente de Silhueta:** Quanto **maior**, melhor. Mede quão bem separados os clusters estão.
    - **Índice de Davies-Bouldin:** Quanto **menor**, melhor. Mede a similaridade média entre cada cluster e seu cluster mais semelhante.
    """)

with tab4:
    st.subheader("Análise Detalhada dos Perfis - K-Means")
    st.markdown(f"Analisando os perfis para **K = {k_otimo}** clusters.")

    labels_kmeans = clustering_models.aplicar_kmeans(df_padronizado, n_clusters=k_otimo)
    perfil_clusters = evaluation.analisar_perfis_clusters(df_numerico, labels_kmeans, 'KMeans')

    st.subheader("Perfil Médio de Cada Cluster")
    st.dataframe(perfil_clusters.style.background_gradient(cmap='viridis', axis=0))

    st.subheader("Visualização dos Perfis (Radar Chart)")
    fig_radar = visualization.plotar_radar_chart(perfil_clusters)
    st.pyplot(fig_radar)
    
    st.info("""
    **Como interpretar o gráfico de radar:**
    - Cada eixo representa uma característica do cliente (dívida, atraso, etc.).
    - Cada linha colorida representa um cluster.
    - O gráfico mostra o 'formato' de cada segmento. Por exemplo, um cluster pode ser 'forte' em `valor_divida_total` e `dias_atraso`, indicando um perfil de alto risco.
    """)

with tab5:
    st.subheader("Sobre este Projeto")
    st.markdown("""
    Este dashboard foi desenvolvido como parte do Trabalho de Conclusão de Curso do MBA em Data Science & Analytics.

    **Objetivo:** Criar uma ferramenta interativa para segmentar clientes inadimplentes, permitindo a análise e comparação de diferentes algoritmos de clusterização para identificar perfis de devedores e otimizar estratégias de negociação.

    **Tecnologias Utilizadas:**
    - **Linguagem:** Python
    - **Bibliotecas Principais:** Streamlit, Pandas, Scikit-learn, Matplotlib, Seaborn
    - **Algoritmos:** K-Means, Clusterização Hierárquica Aglomerativa, DBSCAN
    
    **Autor:** Frederico Antonio Domingues
    """)
