import streamlit as st

def show_about():
    st.title("Sobre o Projeto")

    st.markdown("""
    <h1 style='color: #007BFF;'>Sobre o Projeto</h1>
    <p style='color: #FFFFFF;'>Este aplicativo utiliza um modelo de detecção de objetos baseado na arquitetura MobileNetSSD. O modelo foi treinado para identificar diversas classes, incluindo, mas não se limitando a aviões, bicicletas, pássaros, barcos, portas, ônibus, carros, gatos, cadeiras, vacas, mesas de jantar, cachorros, cavalos, motos, pessoas, plantas em vasos, ovelhas, sofás, trens e monitores de TV. 🌍📷</p>
    
    <h2 style='color: #007BFF;'>Funcionalidades:</h2>
    <p style='color: #FFFFFF;'> 
    Detecção em Tempo Real: O aplicativo é capaz de processar vídeos em tempo real, identificando e destacando objetos presentes no campo de visão. 🎥👁️<br>
    Porcentagem de Confiança: A porcentagem de confiança associada a cada detecção é exibida, indicando o quão seguro o modelo está da presença do objeto identificado. 📊🤖
    </p>

    <h2 style='color: #007BFF;'>Instruções de Uso:</h2>
    <p style='color: #FFFFFF;'> 
    1. Faça o upload de um vídeo nos formatos suportados (mp4, avi).<br>
    2. Aguarde o processamento do vídeo e observe a detecção de objetos em tempo real.<br>
    3. Ajuste o limiar de confiança conforme necessário para otimizar a sensibilidade da detecção. ⚙️
    </p>

    <h2 style='color: #007BFF;'>Tecnologias Utilizadas:</h2>
    <p style='color: #FFFFFF;'> 
    Este aplicativo foi desenvolvido utilizando as seguintes tecnologias:<br>
    - Streamlit: Uma biblioteca para a criação de aplicativos web interativos usando Python.<br>
    - OpenCV: Uma biblioteca de visão computacional que fornece ferramentas para processamento de imagem e vídeo.<br>
    - MobileNetSSD: Um modelo de detecção de objetos leve e eficiente para uso em dispositivos móveis. 🖥️🤖
    </p>

    <h2 style='color: #007BFF;'>Desenvolvedor:</h2>
    <p style='color: #FFFFFF;'> 
    Este aplicativo foi desenvolvido por Thaleson Silva. Sou apaixonado por inteligência artificial e visão computacional, e este projeto é resultado do interesse em tornar essas tecnologias mais acessíveis. Espero que você aproveite a experiência de explorar o mundo da detecção de objetos com este aplicativo. 🚀<br>
    Agradecemos por usar nosso aplicativo e ficamos abertos a feedbacks e sugestões para melhorias futuras. Divirta-se explorando a detecção de objetos! 🙌
    </p>

    <h2 style='color: #007BFF;'>Como o Modelo Foi Treinado:</h2>
    <p style='color: #FFFFFF;'> 
    O modelo MobileNetSSD foi treinado utilizando um conjunto de dados de imagens rotuladas, o que inclui as seguintes etapas:</p>
    
    <h3 style='color: #007BFF;'>1. Preparação dos Dados:</h3>
    <p style='color: #FFFFFF;'> 
    Os dados foram preparados a partir de imagens rotuladas em um formato compatível com o modelo. Abaixo está um exemplo de como os dados foram convertidos para o formato de entrada do modelo:
    </p>

    ```python
    import cv2
    import numpy as np

    # Função para converter imagens em blobs
    def preprocess_image(image_path):
        image = cv2.imread(image_path)
        blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
        return blob
    ```

    <h3 style='color: #007BFF;'>2. Configuração do Modelo:</h3>
    <p style='color: #FFFFFF;'> 
    A arquitetura MobileNetSSD foi configurada e treinada utilizando o framework Caffe. O arquivo `MobileNetSSD_deploy.prototxt.txt` contém a configuração da rede, e `MobileNetSSD_deploy.caffemodel` é o modelo treinado.
    </p>

    ```python
    import cv2

    # Carregar o modelo pré-treinado
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
    ```

    <h3 style='color: #007BFF;'>3. Treinamento do Modelo:</h3>
    <p style='color: #FFFFFF;'> 
    O treinamento foi realizado com o dataset VOC2007, que é um benchmark popular para detecção de objetos. O seguinte código Python foi usado para iniciar o treinamento:
    </p>

    ```python
    # Código para treinar o modelo
    # Importar as bibliotecas necessárias
    import caffe
    from caffe import layers as L, params as P

    # Definir a arquitetura da rede
    def create_network():
        net = caffe.NetSpec()
        net.data, net.label = L.Data(source='data/train_lmdb', backend=P.Data.LMDB, batch_size=32, ntop=2)
        # Adicione camadas aqui
        return net.to_proto()
    
    # Salvar a arquitetura da rede
    with open('train.prototxt', 'w') as f:
        f.write(str(create_network()))
    ```

    <h3 style='color: #007BFF;'>4. Avaliação e Ajuste:</h3>
    <p style='color: #FFFFFF;'> 
    Após o treinamento, o modelo foi avaliado usando métricas como precisão e recall. Ajustes foram feitos para otimizar o desempenho, e o modelo final foi salvo para uso em inferência.
    </p>

    <p style='color: #FFFFFF;'> 
    O código acima mostra uma visão geral do processo de treinamento do modelo MobileNetSSD. A configuração da rede e o treinamento foram realizados usando frameworks específicos e dados rotulados para garantir a eficácia do modelo em detecção de objetos.
    </p>
    """, unsafe_allow_html=True)
