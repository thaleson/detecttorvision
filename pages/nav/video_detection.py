import cv2
import numpy as np
import streamlit as st
import os

def detect_objects(frame, net, confidence_threshold):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1/255.0, (300, 300), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

def show_video_detection():
    st.title("Detecção de Objetos em Vídeo")

    if 'playing' not in st.session_state:
        st.session_state.playing = False
    if 'video_file' not in st.session_state:
        st.session_state.video_file = None
    
    uploaded_file = st.file_uploader("Escolha um arquivo de vídeo", type=["mp4", "avi"])
    if uploaded_file is not None:
        st.session_state.video_file = uploaded_file

    if st.session_state.video_file:
        # Salve o arquivo de vídeo temporariamente
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(st.session_state.video_file.read())
        
        st.write(f"Tentando abrir o arquivo de vídeo: {temp_video_path}")
        
        # Verifique se o arquivo foi salvo corretamente
        if not os.path.exists(temp_video_path):
            st.error(f"Arquivo de vídeo não encontrado no caminho: {temp_video_path}")
            return
        
        cap = cv2.VideoCapture(temp_video_path)
        
        if not cap.isOpened():
            st.error(f"Não foi possível abrir o arquivo de vídeo: {temp_video_path}")
            return
        
        net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')
        confidence_threshold = 0.5

        temp_filename = "temp_output.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(temp_filename, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = detect_objects(frame, net, confidence_threshold)
            out.write(processed_frame)

        cap.release()
        out.release()

        st.write(f"Exibindo o vídeo processado: {temp_filename}")
        st.video(temp_filename)

        try:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except Exception as e:
            st.error(f"Erro ao remover os arquivos temporários: {e}")
