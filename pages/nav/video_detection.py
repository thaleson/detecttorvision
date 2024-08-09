import cv2
import numpy as np
import streamlit as st
import tempfile
import os

def detect_objects(frame, net, confidence_threshold):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = f"{confidence:.2f}"
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

def show_video_detection():
    st.title("Detecção de Objetos em Vídeo")
    
    uploaded_file = st.file_uploader("Escolha um arquivo de vídeo", type=["mp4", "avi"])
    if uploaded_file:
        temp_filename = tempfile.mktemp(suffix=".avi")
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.read())
        
        # Usando os nomes corretos dos arquivos
        prototxt_path = 'MobileNetSSD_deploy.prototxt.txt'
        caffemodel_path = 'MobileNetSSD_deploy.caffemodel'
        
        if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
            st.error(f"Não foi possível encontrar os arquivos do modelo: {prototxt_path} e/ou {caffemodel_path}")
            return
        
        try:
            net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {e}")
            return
        
        confidence_threshold = 0.5
        
        cap = cv2.VideoCapture(temp_filename)
        if not cap.isOpened():
            st.error(f"Não foi possível abrir o arquivo de vídeo: {temp_filename}")
            return
        
        temp_output_filename = tempfile.mktemp(suffix=".avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(temp_output_filename, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
        
        st.write("Processando vídeo...")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Vídeo processado com sucesso.")
                break
            
            processed_frame = detect_objects(frame, net, confidence_threshold)
            out.write(processed_frame)
        
        cap.release()
        out.release()
        
        st.video(temp_output_filename)
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        if os.path.exists(temp_output_filename):
            os.remove(temp_output_filename)

        st.write("Vídeo exibido com sucesso.")
