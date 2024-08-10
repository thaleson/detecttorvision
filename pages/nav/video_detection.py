import os
import tempfile
import streamlit as st
import cv2
import numpy as np

# Carregue o modelo usando OpenCV (Caffe)
def load_model():
    try:
        net = cv2.dnn.readNetFromCaffe(
            'MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
        if net.empty():
            st.error("Erro ao carregar o modelo. Verifique os arquivos do modelo.")
        return net
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Mapeamento de classes
CLASSES = ["fundo", "avião", "bicicleta", "pássaro", "barco",
           "porta", "ônibus", "carro", "gato", "cadeira", "vaca", "mesa de jantar",
           "cachorro", "cavalo", "moto", "pessoa", "planta em vaso", "ovelha",
           "sofá", "trem", "monitor de TV"]

def detect_objects(frame, net, confidence_threshold):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if confidence > confidence_threshold:
            class_name = CLASSES[class_id]
            percentage = confidence * 100
            label = f"{class_name}: {percentage:.2f}%"
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
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
        # Salvar o vídeo carregado em um arquivo temporário
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        with open(temp_input.name, "wb") as f:
            f.write(uploaded_file.read())
        
        # Exibir o vídeo carregado
        st.video(temp_input.name, format="video/mp4", start_time=0)

        # Carregar o modelo
        net = load_model()

        # Verificar se o modelo foi carregado corretamente
        if net is None or net.empty():
            st.error("Modelo não carregado corretamente. Verifique o arquivo do modelo.")
            return

        # Processar o vídeo
        video = cv2.VideoCapture(temp_input.name)
        if not video.isOpened():
            st.error("Não foi possível abrir o vídeo.")
            return
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out = cv2.VideoWriter(temp_output.name, fourcc, 30.0, 
                              (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                               int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while True:
            ret, frame = video.read()
            if not ret:
                break

            result_frame = detect_objects(frame, net, confidence_threshold=0.2)
            out.write(result_frame)

        video.release()
        out.release()

        # Exibir o vídeo processado
        st.video(temp_output.name, format="video/mp4", start_time=0)

        # Remover os arquivos temporários
        try:
            if os.path.exists(temp_input.name):
                os.remove(temp_input.name)
            if os.path.exists(temp_output.name):
                os.remove(temp_output.name)
        except PermissionError:
            st.error("Não foi possível excluir o arquivo temporário. Ele será excluído quando o aplicativo for fechado.")

if __name__ == "__main__":
    show_video_detection()
