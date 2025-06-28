import cv2
import numpy as np

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Color ranges and labels
    color_ranges = {
        "Red": [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([160, 100, 100]), np.array([180, 255, 255]))
        ],
        "Green": [
            (np.array([36, 50, 70]), np.array([89, 255, 255]))
        ],
        "Blue": [
            (np.array([90, 50, 70]), np.array([128, 255, 255]))
        ],
        "Yellow": [
            (np.array([20, 100, 100]), np.array([30, 255, 255]))
        ],
        "Purple": [
            (np.array([129, 50, 70]), np.array([158, 255, 255]))
        ]
    }

    # Drawing colors
    color_bgr = {
        "Red": (0, 0, 255),
        "Green": (0, 255, 0),
        "Blue": (255, 0, 0),
        "Yellow": (0, 255, 255),
        "Purple": (255, 0, 255)
    }

    for color_name, ranges in color_ranges.items():
        # Combine masks for each range
        mask = None
        for lower, upper in ranges:
            current_mask = cv2.inRange(hsv, lower, upper)
            mask = current_mask if mask is None else mask | current_mask

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr[color_name], 2)

                # Draw contour
                cv2.drawContours(frame, [cnt], 0, color_bgr[color_name], 1)

                # Label
                cv2.putText(frame, f"{color_name} Object", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr[color_name], 2)

                # Draw center dot
                cx = x + w // 2
                cy = y + h // 2
                cv2.circle(frame, (cx, cy), 4, color_bgr[color_name], -1)

    # Show results
    cv2.imshow("Multi-Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()