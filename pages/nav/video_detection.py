import cv2
import numpy as np
import streamlit as st
import tempfile
import os
import requests

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

def download_video(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        temp_filename = tempfile.mktemp(suffix=".mp4")
        with open(temp_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return temp_filename
    except requests.RequestException as e:
        st.error(f"Erro ao baixar o vídeo: {e}")
        return None

def show_video_detection():
    st.title("Detecção de Objetos em Vídeo")
    
    url = st.text_input("Cole a URL do vídeo MP4")
    if url:
        temp_filename = download_video(url)
        if not temp_filename:
            return
        
        # Caminhos dos arquivos de modelo
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
        
        # Verificar tamanho do frame
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec para MP4
        temp_output_filename = tempfile.mktemp(suffix=".mp4")
        out = cv2.VideoWriter(temp_output_filename, fourcc, 30.0, (frame_width, frame_height))
        
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
        
        # Exibir vídeo processado
        if os.path.exists(temp_output_filename):
            st.video(temp_output_filename, format="video/mp4")  # Certifique-se de usar o formato correto
        else:
            st.error("O vídeo processado não foi encontrado.")

        # Limpeza de arquivos temporários
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        if os.path.exists(temp_output_filename):
            os.remove(temp_output_filename)

        st.write("Vídeo exibido com sucesso.")
