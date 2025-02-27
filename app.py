from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier('static/haarcascades/haarcascade_frontalface_default.xml')


def analyze_posture(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    alerts = []

    if len(faces) > 0:
        x, y, w, h = faces[0]

        # 1. Center Position Check
        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        face_center = (x + w // 2, y + h // 2)

        x_offset = abs(frame_center[0] - face_center[0])
        y_offset = abs(frame_center[1] - face_center[1])

        if x_offset > frame.shape[1] * 0.2:
            alerts.append("Center your face horizontally")
        if y_offset > frame.shape[0] * 0.2:
            alerts.append("Center your face vertically")

        # 2. Distance Check
        face_area = w * h
        frame_area = frame.shape[0] * frame.shape[1]

        if face_area < frame_area * 0.1:
            alerts.append("Move closer to camera")
        elif face_area > frame_area * 0.3:
            alerts.append("Move back from camera")

        # 3. Tilt Check (Simple aspect ratio)
        aspect_ratio = w / h
        if aspect_ratio < 0.7 or aspect_ratio > 1.3:
            alerts.append("Keep your head straight")

        # Draw visualization
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, frame_center, 5, (0, 0, 255), -1)

    return frame, alerts


def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        processed_frame, alerts = analyze_posture(frame)

        # Add alerts to frame
        if alerts:
            cv2.putText(processed_frame, "Posture Alert!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            for i, alert in enumerate(alerts):
                cv2.putText(processed_frame, alert, (10, 70 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)