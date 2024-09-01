from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
from ultralytics import YOLO
import os
import threading
import time

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Define the classes we are interested in
INTERESTING_CLASSES = ['person', 'cell phone', 'dog', 'cat', 'car', 'motorbike', 'bicycle', 'bus', 'truck']

# Folder to store uploaded images
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def delete_file_after_delay(filepath, delay):
    """Delete a file after a specified delay."""
    time.sleep(delay)
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"Deleted file: {filepath}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Perform object detection
        frame = cv2.imread(filepath)
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

        # Save the resulting image
        result_filepath = os.path.join(UPLOAD_FOLDER, 'result_' + file.filename)
        cv2.imwrite(result_filepath, frame)

        # Start a thread to delete the file after 10 minutes (600 seconds)
        threading.Thread(target=delete_file_after_delay, args=(result_filepath, 600)).start()

        return redirect(url_for('display_result', image_name='result_' + file.filename))

@app.route('/uploads/<image_name>')
def display_result(image_name):
    return send_file(os.path.join(UPLOAD_FOLDER, image_name))

if __name__ == '__main__':
    app.run(debug=True)
