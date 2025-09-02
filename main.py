import streamlit as st
import random

# Define clusters simulados
def definir_cluster(perfil):
    if perfil == "Jovem com baixa renda":
        return {
            "cluster": 1,
            "temperature": 0.9,
            "top_p": 0.95,
            "prompt": "Olá {nome}, tudo bem? Sabemos que imprevistos acontecem. Pensando no seu momento atual, conseguimos uma condição super flexível para te ajudar a quitar sua dívida. Quer ver como é fácil resolver isso agora?"
        }
    elif perfil == "Adulto com renda média":
        return {
            "cluster": 2,
            "temperature": 0.6,
            "top_p": 0.85,
            "prompt": "Prezado {nome}, identificamos uma proposta adequada ao seu perfil para regularização da dívida. Condições acessíveis e possibilidade de parcelamento em até 12x. Clique abaixo e veja como é simples retomar o controle da sua vida financeira."
        }
    elif perfil == "Idoso com perfil conservador":
        return {
            "cluster": 3,
            "temperature": 0.4,
            "top_p": 0.75,
            "prompt": "Senhor(a) {nome}, estamos oferecendo uma proposta personalizada para quitar sua dívida de forma prática e tranquila. Nossa equipe está à disposição para explicar os detalhes. A regularização pode ser feita em poucos passos, com segurança e facilidade."
        }

# Streamlit interface
st.title("MVP - Negociação de Dívidas com LLM Personalizada")

nome = st.text_input("Digite o nome do cliente")
perfil = st.selectbox("Escolha o perfil do cliente", [
    "Jovem com baixa renda",
    "Adulto com renda média",
    "Idoso com perfil conservador"
])

if st.button("Gerar mensagem personalizada"):
    dados = definir_cluster(perfil)
    mensagem = dados["prompt"].format(nome=nome if nome else "Cliente")
    
    st.markdown("---")
    st.subheader(f"Cluster identificado: {dados['cluster']}")
    st.text(f"Parâmetros - temperature: {dados['temperature']}, top_p: {dados['top_p']}")
    st.markdown(f"### Mensagem gerada:\n> {mensagem}")
