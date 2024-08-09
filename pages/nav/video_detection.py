import cv2
import numpy as np
import streamlit as st

def detect_objects(frame, net, confidence_threshold):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1/255.0, (300, 300), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = np.clip(box, 0, [w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = f"{confidence:.2f}"
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame

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
        cap = cv2.VideoCapture(st.session_state.video_file)
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

        st.video(temp_filename)

        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except Exception as e:
            st.error(f"Erro ao remover o arquivo temporário: {e}")

