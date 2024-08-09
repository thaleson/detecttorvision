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

    if 'playing' not in st.session_state:
        st.session_state.playing = False
    if 'video_file' not in st.session_state:
        st.session_state.video_file = None
    if 'processed_file' not in st.session_state:
        st.session_state.processed_file = None

    uploaded_file = st.file_uploader("Escolha um vídeo", type=["mp4", "avi"])

    if uploaded_file is not None:
        # Salvar arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.read())
            temp_filename = tfile.name
            st.session_state.video_file = temp_filename

        # Processar o vídeo e salvar o arquivo processado
        processed_filename = tempfile.mktemp(suffix=".mp4")
        st.session_state.processed_file = processed_filename

        # Processar vídeo com OpenCV
        cap = cv2.VideoCapture(temp_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(processed_filename, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = detect_objects(frame, confidence_threshold=0.2)
            out.write(processed_frame)

        cap.release()
        out.release()

        # Exibir o vídeo processado
        st.video(processed_filename)

        # Remoção do arquivo temporário com verificação de existência
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            if os.path.exists(processed_filename):
                os.remove(processed_filename)
        except PermissionError:
            st.error("Não foi possível excluir o arquivo temporário. Ele será excluído quando o aplicativo for fechado.")

if __name__ == "__main__":
    show_video_detection()
