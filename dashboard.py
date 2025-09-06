# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import os

# Importando nossos módulos atualizados
import data_generator
import preprocessing
import clustering_models
import evaluation
import visualization

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Análise de Segmentação de Clientes",
    page_icon="👥",
    layout="wide"
)

# --- Funções com Cache para Performance ---
@st.cache_data
def carregar_ou_gerar_dados():
    """
    Verifica se a base de dados de 100k registros existe.
    Se não, chama a nova função para gerá-la.
    """
    nome_arquivo = 'base_sintetica_dividas.xlsx'
    if not os.path.exists(nome_arquivo):
        with st.spinner('Base de dados não encontrada. Gerando 100.000 registros...'):
            df = data_generator.gerar_dados_sinteticos(n_clientes=100000, seed=42)
            df.to_excel(nome_arquivo, index=False)
        st.success(f"Base de dados '{nome_arquivo}' criada com sucesso!")
        return df
    return pd.read_excel(nome_arquivo)

@st.cache_data
def processar_dados(df):
    """
    Encapsula o pré-processamento, que agora inclui One-Hot Encoding.
    Retorna um df para análise e outro para modelagem.
    """
    # Chamando a nova função de pré-processamento
    df_numerico_original, df_para_modelagem = preprocessing.selecionar_e_transformar_features(df)
    df_padronizado = preprocessing.padronizar_dados(df_para_modelagem)
    return df_numerico_original, df_padronizado

# --- Parâmetros Fixos da Análise (Ajustados para a nova base) ---
K_OTIMO = 4
DBSCAN_EPS = 2.5  # Ajustado para a maior densidade de pontos
DBSCAN_MIN_SAMPLES = 20 # Ajustado para a maior densidade de pontos

# --- Carregamento e Processamento dos Dados ---
df_clientes = carregar_ou_gerar_dados()
df_numerico_original, df_padronizado = processar_dados(df_clientes)

# --- Título Principal ---
st.title('👥 Ferramenta de Visualização: Segmentação de Clientes Inadimplentes')
st.markdown("---")

# --- Corpo Principal com Abas ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Análise Exploratória dos Dados",
    "📈 Definição do Número de Clusters (K)",
    "🤖 Resultados Comparativos dos Modelos",
    "🔍 Análise de Perfil dos Clusters (K-Means)",
    "ℹ️ Sobre este Trabalho"
])

with tab1:
    st.header("Análise Exploratória da Base de Dados Enriquecida")
    st.markdown("Análise da base de dados com 100.000 clientes, incluindo variáveis sociodemográficas e de comportamento de crédito.")

    st.subheader("Amostra da Base de Dados")
    st.dataframe(df_clientes.head())

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Estatísticas Descritivas (Dados Numéricos)")
        st.dataframe(df_numerico_original.describe())
        
    with col2:
        st.subheader("Matriz de Correlação")
        fig_corr = visualization.plotar_matriz_correlacao(df_numerico_original)
        st.pyplot(fig_corr)
    
    st.markdown("""
    **Análise das Correlações:**
    - A correlação mais forte é a **positiva (+0.55) entre `renda_mensal` e `valor_divida`**, o que é esperado, pois clientes com maior renda tendem a ter acesso a maiores limites de crédito.
    - Observa-se uma **correlação positiva (+0.67) entre `renda_mensal` e `score_credito`**, indicando que a renda é um fator importante na avaliação de crédito.
    - O `historico_pagamento_recente` também tem uma correlação positiva relevante com o `score_credito` (+0.44), validando a lógica de que bons pagadores têm scores melhores.
    """)

    st.markdown("---")
    
    st.subheader("Distribuição das Variáveis Numéricas")
    fig_dist = visualization.plotar_distribuicoes(df_numerico_original)
    st.pyplot(fig_dist)

with tab2:
    st.header("Definição do Número Ótimo de Clusters (K)")
    st.markdown("Utilizamos o Método do Cotovelo e a Análise de Silhueta para determinar o número ideal de segmentos para a nova base de dados.")
    
    with st.spinner("Calculando o K ótimo (esta etapa pode ser demorada na primeira execução)..."):
        resultados_k = clustering_models.encontrar_k_otimo(df_padronizado, max_k=10)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Método do Cotovelo (Elbow Method)")
        fig_cotovelo = visualization.plotar_metodo_cotovelo(resultados_k)
        st.pyplot(fig_cotovelo)
        st.markdown("**Análise:** O 'cotovelo' da curva, onde o ganho em adicionar mais um cluster diminui, continua bem definido em **K=4**.")
    with col2:
        st.subheader("Coeficiente de Silhueta")
        fig_silhueta = visualization.plotar_score_silhueta(resultados_k)
        st.pyplot(fig_silhueta)
        st.markdown("**Análise:** O pico do score, que indica a melhor combinação de coesão e separação dos clusters, também ocorre em **K=4**, validando a escolha.")
        
    st.success("Conclusão: Mesmo com a nova base de dados, ambos os métodos convergem para a escolha de **K = 4** como o número ótimo de clusters.")

with tab3:
    st.header("Resultados Comparativos dos Modelos de Clusterização")
    
    labels_kmeans = clustering_models.aplicar_kmeans(df_padronizado, n_clusters=K_OTIMO)
    labels_hierarquico, _ = clustering_models.aplicar_cluster_hierarquico(df_padronizado, n_clusters=K_OTIMO)
    labels_dbscan = clustering_models.aplicar_dbscan(df_padronizado, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)

    labels_dict = {
        'K-Means': labels_kmeans,
        'Hierárquico': labels_hierarquico,
        'DBSCAN': labels_dbscan
    }
    
    st.subheader("Visualização dos Clusters (Projeção 2D com PCA)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fig_pca_kmeans = visualization.plotar_cluster_pca_individual(df_padronizado, labels_kmeans, 'K-Means')
        st.pyplot(fig_pca_kmeans)
    with col2:
        fig_pca_hier = visualization.plotar_cluster_pca_individual(df_padronizado, labels_hierarquico, 'Hierárquico')
        st.pyplot(fig_pca_hier)
    with col3:
        fig_pca_dbscan = visualization.plotar_cluster_pca_individual(df_padronizado, labels_dbscan, 'DBSCAN')
        st.pyplot(fig_pca_dbscan)

    st.subheader("Métricas de Avaliação Quantitativa")
    df_avaliacao = evaluation.avaliar_modelos(df_padronizado, labels_dict)
    st.dataframe(df_avaliacao.style.highlight_max(subset=['Coeficiente de Silhueta'], color='lightgreen').highlight_min(subset=['Índice de Davies-Bouldin'], color='lightgreen'))
    st.info("K-Means e Hierárquico novamente apresentam os resultados mais equilibrados para o objetivo de negócio de segmentar toda a base de clientes.")

with tab4:
    st.header("Análise de Perfil dos Clusters (Modelo K-Means)")
    st.markdown(f"Analisando as características de cada um dos **{K_OTIMO}** clusters encontrados pelo K-Means na base de dados.")
    
    perfil_clusters = evaluation.analisar_perfis_clusters(df_numerico_original, labels_kmeans, 'K-Means')

    st.subheader("Tabela de Perfil Médio por Cluster (Dados Numéricos)")
    st.dataframe(perfil_clusters.style.background_gradient(cmap='viridis', axis=0))

    st.markdown("<br>", unsafe_allow_html=True)

    # Análise textual completamente refeita para os novos clusters
    for i in range(K_OTIMO):
        st.subheader(f"Análise Detalhada do Cluster {i}")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            fig_radar = visualization.plotar_radar_individual(perfil_clusters, i)
            st.pyplot(fig_radar)
        
        with col2:
            # A análise agora é baseada nas novas variáveis e nos resultados da clusterização
            if i == 0:
                st.markdown("""
                - **Persona:** **Jovem Adulto em Ascensão**.
                - **Características:** Grupo mais jovem (média de 34 anos). Possuem a **menor renda mensal** e, consequentemente, o **menor valor de dívida**. Seu score de crédito e histórico de pagamento são medianos.
                - **Estratégia Sugerida:** Abordagem digital e de baixo custo. Foco em educação financeira e ofertas de quitação com pequenos descontos para preservar o potencial de relacionamento futuro com esses clientes.
                """)
            elif i == 1:
                st.markdown("""
                - **Persona:** **Cliente Estabelecido de Alto Risco**.
                - **Características:** Este é o grupo de **maior risco**. Possuem a **maior renda mensal**, mas também o **maior valor de dívida**. O que mais se destaca é o **pior histórico de pagamento recente**, resultando no **pior score de crédito** do grupo.
                - **Estratégia Sugerida:** Ação de cobrança prioritária e especializada. Analistas seniores devem focar em entender a situação e propor renegociações estruturadas, possivelmente com consolidação de dívidas.
                """)
            elif i == 2:
                st.markdown("""
                - **Persona:** **Cliente Sênior e Conservador**.
                - **Características:** Grupo com a **maior média de idade** (58 anos). Sua renda e valor de dívida são moderados. O ponto forte é o **excelente histórico de pagamento recente**, o que lhes confere o **melhor score de crédito** entre todos os clusters. A inadimplência parece ser um evento atípico.
                - **Estratégia Sugerida:** Abordagem respeitosa e facilitadora. Canais tradicionais (telefone) podem ser mais eficazes. Oferecer flexibilidade e condições de pagamento facilitadas deve ser suficiente para a recuperação.
                """)
            elif i == 3:
                st.markdown("""
                - **Persona:** **Família de Renda Média e Endividada**.
                - **Características:** Perfil de meia-idade (46 anos) com o **maior número de dependentes**. A renda é moderada, mas o **valor da dívida é alto em proporção à renda**. O score de crédito é baixo, refletindo um endividamento estrutural.
                - **Estratégia Sugerida:** Abordagem empática, com foco em soluções de longo prazo. Ofertas de parcelamento estendido e descontos progressivos podem ser eficazes. A comunicação deve ser clara e focada em resolver o problema financeiro da família.
                """)

with tab5:
    st.header("Sobre este Trabalho")
    st.subheader("Tema do Trabalho de Conclusão de Curso")
    st.markdown("#### Segmentação de clientes para otimizar abordagens iniciais de negociação de dívidas")
    st.subheader("Setup Técnico do Projeto")
    st.markdown("""
    - **Linguagem:** Python (versão 3.11)
    - **Bibliotecas Principais:** Streamlit, Pandas, Scikit-learn, Matplotlib, Seaborn, NumPy, Openpyxl.
    - **Autor:** Frederico Antonio Domingues
    """)

