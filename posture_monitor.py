import cv2
import numpy as np
import time

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


# Sound alert (optional, requires `simpleaudio` or `playsound`)
def play_alert():
    try:
        import simpleaudio as sa
        wave_obj = sa.WaveObject.from_wave_file("alert.wav")
        wave_obj.play()
    except ImportError:
        print("Install `simpleaudio` for sound alerts.")


# Posture analysis function
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


# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Analyze posture
    processed_frame, alerts = analyze_posture(frame)

    # Display alerts
    if alerts:
        cv2.putText(processed_frame, "Posture Alert!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        for i, alert in enumerate(alerts):
            cv2.putText(processed_frame, alert, (10, 70 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        play_alert()  # Play sound alert

    # Show the frame
    cv2.imshow("Posture Monitor", processed_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()