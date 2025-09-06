import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Dashboard - Avaliação de Mensagens A/B")

# Upload dos dados do Forms
arquivo = st.file_uploader("Envie o CSV de respostas do Google Forms", type="csv")

if arquivo:
    df = pd.read_csv(arquivo)

    # Renomear colunas para facilitar (ajustar conforme o Forms criado)
    df.columns = [
        "timestamp", "perfil", "grupo", "clareza", "empatia", "resposta"
    ]

    st.subheader("Amostra de Respostas")
    st.dataframe(df.head())

    # Agrupar e calcular médias
    medias = df.groupby(["perfil", "grupo"])[["clareza", "empatia", "resposta"]].mean().reset_index()

    st.subheader("Médias por Perfil e Grupo")
    st.dataframe(medias)

    # Gráfico comparativo
    st.subheader("Comparativo Visual")
    metricas = ["clareza", "empatia", "resposta"]

    for metrica in metricas:
        st.markdown(f"### {metrica.title()} - Grupo A vs B")
        fig, ax = plt.subplots()
        sns.barplot(data=medias, x="perfil", y=metrica, hue="grupo", palette="Set2", ax=ax)
        ax.set_ylabel(f"Média de {metrica}")
        ax.set_xlabel("Perfil")
        ax.set_title(f"{metrica.title()} por Perfil e Grupo")
        st.pyplot(fig)
else:
    st.info("Envie um arquivo CSV com os resultados coletados no Google Forms para visualizar o dashboard.")
