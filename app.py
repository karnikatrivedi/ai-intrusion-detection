import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load the pre-trained YOLO AI model
model = YOLO("yolov8n.pt")

# 2. Access your webcam (0 is default webcam)
cap = cv2.VideoCapture(0)

# 3. Define the restricted red boundary zone (Box coordinates on screen)
zone_points = np.array([[100, 100], [400, 100], [400, 400], [100, 400]], np.int32)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Detect humans (class 0 is person)
    results = model(frame, classes=[0], verbose=False)

    # Draw the red restricted zone box
    cv2.polylines(frame, [zone_points], isClosed=True, color=(0, 0, 255), thickness=3)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Draw green dot at the center of the person
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

        # Check if the person's center point is inside the red zone
        is_inside = cv2.pointPolygonTest(zone_points, (center_x, center_y), False)

        if is_inside >= 0:
            # INTRUSION DETECTED: Draw red box & warning text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame, "WARNING: INTRUSION DETECTED!", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            # SAFE: Draw green box around person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display video feed window
    cv2.imshow("Smart AI Security Guard", frame)

    # Press 'q' on keyboard to exit video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
