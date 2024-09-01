from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Define the classes we are interested in
INTERESTING_CLASSES = ['person', 'cell phone', 'dog', 'cat', 'car', 'motorbike', 'bicycle', 'bus', 'truck']

def generate_frames():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video file path
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Perform object detection
        results = model(frame)

        # Iterate through results and draw bounding boxes with labels
        for result in results:
            boxes = result.boxes  # List of detected boxes
            for box in boxes:
                class_id = int(box.cls[0])
                label = result.names[class_id]
                confidence = box.conf[0]

                if label in INTERESTING_CLASSES:
                    # Extract box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Define colors for different categories
                    color = (0, 255, 0)  # Default green for other objects
                    if label == 'person':
                        color = (255, 0, 0)  # Blue for humans
                    elif label in ['car', 'motorbike', 'bicycle', 'bus', 'truck']:
                        color = (0, 0, 255)  # Red for vehicles
                    elif label in ['dog', 'cat']:
                        color = (0, 255, 255)  # Yellow for animals
                    elif label == 'cell phone':
                        color = (255, 255, 0)  # Cyan for mobiles

                    # Draw the bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as an HTTP response with multipart MIME type
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
