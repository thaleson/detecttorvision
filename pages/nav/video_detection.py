import os
import tempfile
import streamlit as st
import cv2
import numpy as np
import time

# Carregue o modelo usando OpenCV (Caffe)
def load_model():
    return cv2.dnn.readNetFromCaffe(
        'MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# Mapeamento de classes
CLASSES = ["fundo", "avião", "bicicleta", "pássaro", "barco",
           "porta", "ônibus", "carro", "gato", "cadeira", "vaca", "mesa de jantar",
           "cachorro", "cavalo", "moto", "pessoa", "planta em vaso", "ovelha",
           "sofá", "trem", "monitor de TV"]

def detect_objects(frame, net, confidence_threshold):
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if confidence > confidence_threshold:
            class_name = CLASSES[class_id]
            percentage = confidence * 100
            label = f"{class_name}: {percentage:.2f}%"
            box = detections[0, 0, i, 3:7] * np.array(
                [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

def initialize_video(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.read())
        temp_filename = tfile.name
    return temp_filename

def show_video_detection():
    st.title("Detecção de Objetos em Vídeo🕵️‍♂🎥")

    if 'video_file' not in st.session_state:
        st.session_state.video_file = None

    # Aviso sobre as limitações do modelo
    st.warning("Aviso: O modelo MobileNetSSD pode não detectar todos os objetos em vídeos e é limitado a vídeos apenas.")

    uploaded_file = st.file_uploader("Escolha um vídeo", type=["mp4", "avi"])

    if uploaded_file:
        st.session_state.video_file = initialize_video(uploaded_file)

        # Carregar o modelo
        net = load_model()

        # Exibir o vídeo
        st.video(st.session_state.video_file, format="video/mp4", start_time=0)

        # Processar o vídeo em segundo plano
        def process_video():
            video = cv2.VideoCapture(st.session_state.video_file)
            frame_rate = video.get(cv2.CAP_PROP_FPS)

            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break

                result_frame = detect_objects(frame, net, confidence_threshold=0.2)

                # Exibir o frame processado
                st.image(result_frame, channels="BGR", use_column_width=True)

                # Ajustar a reprodução de acordo com a velocidade selecionada
                wait_time = (1 / frame_rate)  # Velocidade fixa
                time.sleep(wait_time)

            video.release()

        # Executar o processamento em segundo plano
        st.spinner("Processando o vídeo...")
        process_video()

        # Remoção do arquivo temporário com verificação de existência
        try:
            if os.path.exists(st.session_state.video_file):
                os.remove(st.session_state.video_file)
        except PermissionError:
            st.error("Não foi possível excluir o arquivo temporário. Ele será excluído quando o aplicativo for fechado.")

if __name__ == "__main__":
    show_video_detection()
