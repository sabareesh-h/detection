"""
=============================================================
  webcam_demo.py  --  Defect detection pipeline script
=============================================================
HOW TO USE
----------
python webcam_demo.py

FLAGS
-----
(No flags defined or --help not available)
=============================================================
"""

import cv2
from ultralytics import YOLO, RTDETR

# Load the trained model
model_path = r"runs/detect/models/defect_gpu_v3/weights/best.pt"
print(f"Loading model from: {model_path}")
model = RTDETR(model_path) if 'rtdetr' in model_path.lower() else YOLO(model_path)

# Initialize webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting live inference... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run inference on the frame
    results = model(frame, verbose=False)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv11 Live Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
