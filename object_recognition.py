# ============================================
# OpenCV Object Recognition using YOLOv8
# Summer Training Project
# ============================================

# -------------------------
# Import Libraries
# -------------------------
import cv2
from ultralytics import YOLO

# -------------------------
# Load YOLOv8 Model
# -------------------------
model = YOLO("yolov8n.pt")

# -------------------------
# Open Webcam
# -------------------------
cap = cv2.VideoCapture(0)

# Check if the camera is available
if not cap.isOpened():
    print("Error: Unable to open the webcam.")
    exit()

print("Camera started successfully.")
print("Press Q to exit.")

# -------------------------
# Start Object Recognition
# -------------------------
while True:

    # Read frame from webcam
    ret, frame = cap.read()

    # Stop if frame cannot be read
    if not ret:
        print("Failed to read frame.")
        break

    # Detect objects
    results = model(frame)

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # Display result
    cv2.imshow("OpenCV Object Recognition", annotated_frame)

    # Exit when pressing Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------------
# Release Resources
# -------------------------
cap.release()
cv2.destroyAllWindows()