import os
import tempfile
import streamlit as st
import cv2
import numpy as np
import time

# Carregue o modelo usando OpenCV (Caffe)
net = cv2.dnn.readNetFromCaffe(
    'MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# Mapeamento de classes
CLASSES = ["fundo", "avião", "bicicleta", "pássaro", "barco",
           "porta", "ônibus", "carro", "gato", "cadeira", "vaca", "mesa de jantar",
           "cachorro", "cavalo", "moto", "pessoa", "planta em vaso", "ovelha",
           "sofá", "trem", "monitor de TV"]

def detect_objects(frame, confidence_threshold):
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

def show_video_detection():
    st.title("Detecção de Objetos em Vídeo")

    # Inicialização do estado da sessão
    if 'playing' not in st.session_state:
        st.session_state.playing = False
    if 'video_status' not in st.session_state:
        st.session_state.video_status = ""
    if 'speed' not in st.session_state:
        st.session_state.speed = 1.0

    # Aviso sobre as limitações do modelo
    st.warning("Aviso: O modelo MobileNetSSD pode não detectar todos os objetos em vídeos e é limitado a vídeos apenas.")

    uploaded_file = st.file_uploader("Escolha um vídeo", type=["mp4", "avi"])

    if uploaded_file is not None:
        st.write("Processando vídeo...")
        
        # Cria um arquivo temporário de forma segura
        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            tfile.write(uploaded_file.read())
            temp_filename = tfile.name  # Salva o nome do arquivo temporário

        video = cv2.VideoCapture(temp_filename)
        stframe = st.empty()

        # Controles de vídeo alinhados com ícones
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶️ Play"):
                st.session_state.playing = True
                st.session_state.video_status = "Vídeo em reprodução..."
        with col2:
            if st.button("⏸️ Pausar"):
                st.session_state.playing = False
                st.session_state.video_status = "Vídeo pausado."
        with col3:
            if st.button("⏹️ Parar"):
                st.session_state.playing = False
                st.session_state.video_status = "Vídeo parado."
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Volta ao início do vídeo

        # Controles de velocidade
        st.write("Controle de velocidade:")
        col4, col5, col6 = st.columns(3)
        with col4:
            if st.button("0.5x"):
                st.session_state.speed = 0.5
                st.session_state.video_status = f"Velocidade ajustada para {st.session_state.speed}x."
        with col5:
            if st.button("1x"):
                st.session_state.speed = 1.0
                st.session_state.video_status = "Velocidade ajustada para 1x."
        with col6:
            if st.button("1.5x"):
                st.session_state.speed = 1.5
                st.session_state.video_status = f"Velocidade ajustada para {st.session_state.speed}x."
        with col4:
            if st.button("2x"):
                st.session_state.speed = 2.0
                st.session_state.video_status = f"Velocidade ajustada para {st.session_state.speed}x."

        st.write(st.session_state.video_status)

        frame_rate = video.get(cv2.CAP_PROP_FPS)

        while video.isOpened():
            if st.session_state.playing:
                ret, frame = video.read()
                if not ret:
                    break

                result_frame = detect_objects(frame, confidence_threshold=0.2)
                stframe.image(result_frame, channels="BGR", use_column_width=True)

                # Ajusta a reprodução de acordo com a velocidade selecionada
                wait_time = 1 / (frame_rate * st.session_state.speed)

                # Evita que o loop bloqueie a execução
                if wait_time > 0:
                    time.sleep(wait_time)
            else:
                time.sleep(0.1)  # Evita o uso excessivo de CPU quando o vídeo está pausado

        video.release()

        # Tratamento para o erro PermissionError
        try:
            os.remove(temp_filename)
        except PermissionError:
            st.error("Não foi possível excluir o arquivo temporário. Ele será excluído quando o aplicativo for fechado.")

if __name__ == "__main__":
    show_video_detection()
