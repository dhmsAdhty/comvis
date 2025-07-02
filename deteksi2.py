import cv2
import time
from collections import defaultdict
from ultralytics import YOLO

class ObjectDetectionSystem:
    def __init__(self):
        # Inisialisasi model YOLO dan kamera
        self.model = YOLO('yolov8n.pt')
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Daftar kelas target yang ingin dideteksi beserta warna dan counter
        self.target_classes = {
            0: {'name': 'Person', 'color': (0, 255, 0), 'count': 0},
            67: {'name': 'Phone', 'color': (255, 0, 0), 'count': 0},
            63: {'name': 'Laptop', 'color': (0, 0, 255), 'count': 0},
            62: {'name': 'TV', 'color': (255, 255, 0), 'count': 0},
            66: {'name': 'Keyboard', 'color': (0, 255, 255), 'count': 0},
            64: {'name': 'Mouse', 'color': (255, 0, 255), 'count': 0}
        }

        # Variabel untuk performa dan histori deteksi
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.detection_history = defaultdict(list)

    def draw_dashboard(self, frame):
        """Menampilkan FPS dan jumlah deteksi tiap kelas pada frame."""
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset = 60
        for data in self.target_classes.values():
            cv2.putText(frame, f"{data['name']}: {data['count']}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, data['color'], 1)
            y_offset += 25

    def update_detection_history(self, results):
        """Menyimpan waktu deteksi untuk tiap kelas (history 5 menit terakhir)."""
        current_time = time.time()
        for obj in results[0].boxes:
            class_id = int(obj.cls)
            if class_id in self.target_classes:
                self.detection_history[class_id].append(current_time)
                # Hanya simpan histori 5 menit terakhir
                self.detection_history[class_id] = [
                    t for t in self.detection_history[class_id] if current_time - t < 300
                ]

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Reset counter deteksi tiap kelas
            for data in self.target_classes.values():
                data['count'] = 0

            # --- INFERENSI: Deteksi objek pada frame ---
            results = self.model.predict(
                frame,
                conf=0.6,
                classes=list(self.target_classes.keys())  # --- FILTERING: Hanya kelas target ---
            )

            # --- FILTERING: Hitung hanya objek dengan class_id target ---
            for obj in results[0].boxes:
                class_id = int(obj.cls)
                if class_id in self.target_classes:
                    self.target_classes[class_id]['count'] += 1

            # Update histori deteksi
            self.update_detection_history(results)

            # Hitung FPS
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed

            # Visualisasi hasil deteksi dan dashboard
            annotated_frame = results[0].plot()
            self.draw_dashboard(annotated_frame)

            # Tampilkan hasil
            cv2.imshow('Deteksi Person & Electronic', annotated_frame)

            # Keluar jika tombol 'q' ditekan
            if cv2.waitKey(1) == ord('q'):
                break

        # Bersihkan resource
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetectionSystem()
    detector.run()