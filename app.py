import os
import tempfile
import streamlit as st
import cv2
import numpy as np

# Carregue o modelo usando OpenCV (Caffe)
net = cv2.dnn.readNetFromCaffe(
    'MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# Mapeamento de classes
CLASSES = ["fundo", "avião", "bicicleta", "pássaro", "barco",
           "porta", "ônibus", "carro", "gato", "cadeira", "vaca", "mesa de jantar",
           "cachorro", "cavalo", "moto", "pessoa", "planta em vaso", "ovelha",
           "sofá", "trem", "monitor de TV"]

# Função para realizar a detecção de objetos


def detect_objects(frame):
    # Pré-processamento da imagem conforme necessário
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    # Faça a previsão usando o modelo
    detections = net.forward()

    # Pós-processamento das previsões conforme necessário
    # ...

    # Desenhe caixas delimitadoras, rótulos, etc., no frame
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if confidence > 0.2:  # Ajuste o limiar de confiança conforme necessário
            # Obtenha o nome da classe
            class_name = CLASSES[class_id]

            # Obtenha a porcentagem de confiança
            percentage = confidence * 100

            # Adicione o nome da classe e a porcentagem de confiança à imagem
            label = f"{class_name}: {percentage:.2f}%"
            box = detections[0, 0, i, 3:7] * np.array(
                [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def main():
    st.title("DectetoVision: Descubra o Que os Seus Vídeos Escondem 🕵️‍♂️🎥✨")

    uploaded_file = st.file_uploader("Escolha um vídeo", type=["mp4", "avi"])

    if uploaded_file is not None:
        # Salve o arquivo temporário para um local específico
        video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
        with open(video_path, "wb") as video_file:
            video_file.write(uploaded_file.read())

        # Crie o objeto VideoCapture com o caminho do arquivo
        video = cv2.VideoCapture(video_path)

        # Exiba o vídeo frame a frame com detecção de objetos
        while True:
            ret, frame = video.read()
            if not ret:
                break  # Se não houver mais frames, saia do loop

            # Realize a detecção de objetos no frame
            result_frame = detect_objects(frame)

            # Exiba o resultado em uma janela OpenCV
            cv2.imshow("Detecção de Objetos", result_frame)

            # Aguarde 1 milissegundo (permite que a janela OpenCV seja atualizada)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Se pressionar 'q', saia do loop

        # Limpeza: Feche o objeto VideoCapture, feche a janela OpenCV e exclua o arquivo temporário
        video.release()
        cv2.destroyAllWindows()
        os.remove(video_path)

    if st.button("Sobre este app"):
     st.text(
    """Sobre o Aplicativo de Detecção de Objetos:

Este aplicativo utiliza um modelo de detecção de objetos baseado na arquitetura MobileNetSSD. O modelo foi treinado para identificar diversas classes, incluindo, mas não se limitando a aviões, bicicletas, pássaros, barcos, portas, ônibus, carros, gatos, cadeiras, vacas, mesas de jantar, cachorros, cavalos, motos, pessoas, plantas em vasos, ovelhas, sofás, trens e monitores de TV. 🌍📷

Funcionalidades:

Detecção em Tempo Real: O aplicativo é capaz de processar vídeos em tempo real, identificando e destacando objetos presentes no campo de visão. 🎥👁️

Porcentagem de Confiança: A porcentagem de confiança associada a cada detecção é exibida, indicando o quão seguro o modelo está da presença do objeto identificado. 📊🤖

Instruções de Uso:

1. Faça o upload de um vídeo nos formatos suportados (mp4, avi).
2. Aguarde o processamento do vídeo e observe a detecção de objetos em tempo real.
3. Ajuste o limiar de confiança conforme necessário para otimizar a sensibilidade da detecção. ⚙️

Tecnologias Utilizadas:

Este aplicativo foi desenvolvido utilizando as seguintes tecnologias:

- Streamlit: Uma biblioteca para a criação de aplicativos web interativos usando Python.
- OpenCV: Uma biblioteca de visão computacional que fornece ferramentas para processamento de imagem e vídeo.
- MobileNetSSD: Um modelo de detecção de objetos leve e eficiente para uso em dispositivos móveis. 🖥️🤖

Desenvolvedor:

Este aplicativo foi desenvolvido por Thaleson Silva. Sou apaixonado por inteligência artificial e visão computacional, e este projeto é resultado do interesse em tornar essas tecnologias mais acessíveis. Espero que você aproveite a experiência de explorar o mundo da detecção de objetos com este aplicativo. 🚀

Agradecemos por usar nosso aplicativo e ficamos abertos a feedbacks e sugestões para melhorias futuras. Divirta-se explorando a detecção de objetos! 🙌""")



if __name__ == "__main__":
    main()
