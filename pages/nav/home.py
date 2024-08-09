import streamlit as st
import json
from streamlit_lottie import st_lottie

def show_home():


    # Título principal
    st.title("Bem-vindo ao DetectoVision 🕵️‍♂️🎥! ")

    # Subtítulo
    st.subheader("Olá! Eu sou Thaleson Silva 👋")
    # Colunas que organizam a página
    col1, col2 = st.columns(2)

    # Animações
    with open("animaçoes/pagina_inicial1.json") as source:
        animacao_1 = json.load(source)

    with open("animaçoes/pagina_inicial2.json") as source:
        animacao_2 = json.load(source)
    
    # Conteúdo a ser exibido na coluna 1
    with col1:
        st_lottie(animacao_1, height=450, width=450)
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.markdown("""
            <h5 style='text-align: justify;  line-height: 1.6;'>
                Este projeto de Detecção de Objetos em Vídeo é uma aplicação interativa que utiliza visão computacional para identificar e marcar objetos em vídeos. O objetivo é demonstrar a capacidade do modelo MobileNetSSD em detectar objetos em tempo real e fornecer uma interface de usuário amigável para essa funcionalidade.
            </h5>
        """, unsafe_allow_html=True)

    # Conteúdo a ser exibido na coluna 2
    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.markdown("""
            <h5 style='text-align: justify;  line-height: 1.6;'>
                Bem-vindo ao sistema de Detecção de Objetos em Vídeo! 🎥🚀
                Aqui, você pode fazer o upload de vídeos e visualizar em tempo real a detecção de vários objetos. Aproveite a tecnologia de visão computacional para analisar o conteúdo dos seus vídeos de maneira eficiente e interativa.
            </h5>
        """, unsafe_allow_html=True)
        st_lottie(animacao_2, height=400, width=440)
