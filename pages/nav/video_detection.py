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
           "porta", "ônibus", "carro", "gato", "cadeira", "cavalo", "mesa de jantar",
           "cachorro", "moto", "pessoa", "planta em vaso", "ovelha",
           "sofá", "trem", "monitor de TV"]

def detect_objects(frame, confidence_threshold):
    frame = cv2.resize(frame, (600, 400))  # Ajuste a resolução conforme necessário
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
        st.session_state.playing = True
    if 'video_status' not in st.session_state:
        st.session_state.video_status = ""

    # Aviso sobre as limitações do modelo
    st.warning("Aviso: O modelo MobileNetSSD pode não detectar todos os objetos em vídeos e é limitado a vídeos apenas.")

    uploaded_file = st.file_uploader("Escolha um vídeo", type=["mp4", "avi"])

    if uploaded_file is not None:
        st.write("Processando vídeo...")

        # Criação de um arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tfile:
            tfile.write(uploaded_file.read())
            temp_file_path = tfile.name
            st.write(f"Arquivo temporário criado em: {temp_file_path}")

        try:
            video = cv2.VideoCapture(temp_file_path)
            stframe = st.empty()

            # Ajuste da taxa de quadros
            frame_rate = video.get(cv2.CAP_PROP_FPS)
            frame_time = 1.0 / frame_rate  # Tempo para exibir cada frame

            while video.isOpened():
                if st.session_state.playing:
                    ret, frame = video.read()
                    if not ret:
                        st.write("Fim do vídeo.")
                        break

                    result_frame = detect_objects(frame, confidence_threshold=0.2)
                    stframe.image(result_frame, channels="BGR", use_column_width=True)

                    # Ajusta o tempo de exibição dos frames
                    time.sleep(frame_time)
                else:
                    st.write("Vídeo pausado.")

            video.release()

        except Exception as e:
            st.error(f"Ocorreu um erro ao processar o vídeo: {e}")
        finally:
            # Tente remover o arquivo temporário
            try:
                os.remove(temp_file_path)
                st.write(f"Arquivo temporário removido: {temp_file_path}")
            except Exception as e:
                st.error(f"Ocorreu um erro ao remover o arquivo temporário: {e}")

if __name__ == "__main__":
    show_video_detection()
