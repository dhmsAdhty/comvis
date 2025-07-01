import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Konfigurasi Model
model = YOLO('yolov8n.pt')
target_classes = {
    0: 'Person', 
    67: 'Phone', 
    63: 'Laptop', 
    62: 'TV', 
    66: 'Keyboard', 
    64: 'Mouse'
}

# Session State untuk kontrol
if 'stop_key' not in st.session_state:
    st.session_state.stop_key = 'q'
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False

# Tampilan Streamlit
st.title("🔍 Real-Time Object Detection")
st.caption("Deteksi Orang dan Barang Elektronik dengan YOLOv8")

# Sidebar untuk pengaturan
with st.sidebar:
    st.header("Pengaturan")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5)
    
    # Pilih tombol stop
    new_stop_key = st.text_input("Tombol Stop (1 karakter)", value='q', max_chars=1).lower()
    if new_stop_key:
        st.session_state.stop_key = new_stop_key
    st.info(f"Tekan '{st.session_state.stop_key}' di keyboard untuk stop")

# Fungsi deteksi
def detect_objects(frame):
    results = model.predict(frame, conf=conf_threshold, classes=list(target_classes.keys()))
    annotated_frame = results[0].plot()
    return annotated_frame

# Tombol kontrol webcam
col1, col2 = st.columns(2)
with col1:
    if st.button("🎥 Mulai Webcam", type="primary") and not st.session_state.webcam_active:
        st.session_state.webcam_active = True
        st.rerun()

with col2:
    if st.button("⏹️ Stop Webcam") and st.session_state.webcam_active:
        st.session_state.webcam_active = False
        st.rerun()

# Area tampilan webcam
FRAME_WINDOW = st.image([])

if st.session_state.webcam_active:
    cap = cv2.VideoCapture(0)
    stop_pressed = False
    
    while cap.isOpened() and st.session_state.webcam_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal mengakses kamera!")
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_frame = detect_objects(frame_rgb)
        FRAME_WINDOW.image(annotated_frame)
        
        # Stop dengan tombol keyboard
        if cv2.waitKey(1) & 0xFF == ord(st.session_state.stop_key):
            st.session_state.webcam_active = False
            st.rerun()
    
    cap.release()
    cv2.destroyAllWindows()

# Pilihan upload file
st.divider()
uploaded_file = st.file_uploader("Atau upload file", type=["jpg", "png", "jpeg"])
if uploaded_file and uploaded_file.type.startswith('image'):
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    st.image(detect_objects(img_array), caption="Hasil Deteksi")