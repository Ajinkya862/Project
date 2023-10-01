import cv2
import mediapipe as mp
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

class FaceDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Real-time Face Detection and Face Mesh with SPO2")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.capture = cv2.VideoCapture(0)

        self.spo2_values = []  # to store spo2 values

        # Initialize the matplotlib figure for real-time plotting
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('SPO2')
        self.ax.set_title('Real-time SPO2 Calculation')
        self.ax.set_ylim([90, 100])

        self.mpl_timer = self.canvas.new_timer(interval=30)
        self.mpl_timer.add_callback(self.update_plot)
        self.mpl_timer.start()

    def calculate_spo2(self, frame):
        # Split the frame into its RGB channels
        blue_channel, green_channel, red_channel = cv2.split(frame)

        # Calculate the mean and standard deviation for the red channel
        red_mean = np.mean(red_channel)
        red_std = np.std(red_channel)

        # Calculate the mean and standard deviation for the blue channel
        blue_mean = np.mean(blue_channel)
        blue_std = np.std(blue_channel)

        # Calculate the correlation coefficient (R)
        R = red_mean / blue_mean

        spo2 = 125-28*R
        self.spo2_values.append(spo2)
        return spo2

    def update_plot(self):
        frame_numbers = range(len(self.spo2_values))
        self.ax.clear()
        self.ax.plot(frame_numbers, self.spo2_values)
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('SPO2')
        self.ax.set_title('Real-time SPO2 Calculation')
        self.ax.set_ylim([90, 100])
        self.canvas.draw()

    def update_frame(self):
        ret, frame = self.capture.read()

        if ret:
            # to perform face detection
            results_detection = self.face_detection.process(frame)
            if results_detection.detections:
                for detection in results_detection.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Calculating the ROI coordinates for forehead
                    forehead_x = x
                    forehead_y = y
                    forehead_w = w
                    forehead_h = int(h * 0.25)

                    # Extract the forehead ROI
                    forehead_roi = frame[forehead_y:forehead_y + forehead_h, forehead_x:forehead_x + forehead_w]

                    # Calculate SpO2 for the forehead ROI
                    spo2_forehead = self.calculate_spo2(forehead_roi)

                    # Display SpO2 value on the frame
                    cv2.putText(frame, f'Forehead SpO2: {spo2_forehead:.2f}', (forehead_x, forehead_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # face mesh
                    results_mesh = self.face_mesh.process(frame)
                    if results_mesh.multi_face_landmarks:
                        for face_landmarks in results_mesh.multi_face_landmarks:
                            for landmark in face_landmarks.landmark:
                                x, y = int(landmark.x * iw), int(landmark.y * ih)
                                cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

            h, w, c = frame.shape
            bytes_per_line = c * w
            q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceDetectionApp()
    window.show()
    sys.exit(app.exec_())
