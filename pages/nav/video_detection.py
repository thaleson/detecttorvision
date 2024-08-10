import os
import tempfile
import time
import streamlit as st
import cv2
import numpy as np

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

def show_video_detection():
    st.title("Detecção de Objetos em Vídeo🕵️‍♂🎥")

    if 'video_file' not in st.session_state:
        st.session_state.video_file = None

    uploaded_file = st.file_uploader("Escolha um vídeo", type=["mp4", "avi"])

    if uploaded_file:
        st.session_state.video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(st.session_state.video_file, "wb") as f:
            f.write(uploaded_file.read())

        # Exibir o vídeo normalmente
        st.video(st.session_state.video_file, format="video/mp4", start_time=0)

        # Carregar o modelo
        net = load_model()

        # Processar o vídeo em segundo plano e mostrar o vídeo processado
        video = cv2.VideoCapture(st.session_state.video_file)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(temp_output, fourcc, 30.0, (int(video.get(3)), int(video.get(4))))

        while True:
            ret, frame = video.read()
            if not ret:
                break

            result_frame = detect_objects(frame, net, confidence_threshold=0.2)
            out.write(result_frame)
            time.sleep(0.1)  # Pequena pausa para garantir que o modelo processe o frame

        video.release()
        out.release()

        st.video(temp_output, format="video/mp4", start_time=0)

        # Remover o arquivo temporário com verificação de existência
        try:
            if os.path.exists(st.session_state.video_file):
                os.remove(st.session_state.video_file)
            if os.path.exists(temp_output):
                os.remove(temp_output)
        except PermissionError:
            st.error("Não foi possível excluir o arquivo temporário. Ele será excluído quando o aplicativo for fechado.")

if __name__ == "__main__":
    show_video_detection()
