from ultralytics import YOLO
import cv2

# Muat model
model = YOLO('yolov8n.pt')  # Model YOLOv8n

# Buka webcam
cap = cv2.VideoCapture(0)

# Daftar kelas yang dideteksi (person + elektronik)
target_classes = {
    0: 'person',        # Orang
    67: 'cell phone',   # HP
    63: 'laptop',       # Laptop
    62: 'tv',           # TV
    66: 'keyboard',     # Keyboard
    64: 'mouse'         # Mouse
}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Deteksi dengan filter kelas
    results = model.predict(frame, conf=0.5, classes=list(target_classes.keys()))
    
    # Visualisasi hasil
    annotated_frame = results[0].plot()
            
    # Tampilkan daftar objek terdeteksi
    for obj in results[0].boxes:
        class_id = int(obj.cls)
        if class_id in target_classes:
            label = target_classes[class_id]
            conf = float(obj.conf)
            print(f"Deteksi: {label} ({conf:.2f})")
    
    cv2.imshow('Deteksi Orang & Elektronik', annotated_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()