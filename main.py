import cv2
import time

from cnn_detection.facenet_detection import FaceDetectionRecognition

# Open a connection to the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit.")

frame_skip = 2  # Skip every 2 frames
frame_count = 0

facenet = FaceDetectionRecognition()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    start_time = time.time()


    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection
    boxes, _, nfaces = facenet.detect_faces(img_rgb)
    print(nfaces)

    age, gender = facenet.predict_age_gender(facenet.extract_face(img_rgb, boxes))
    print(f"Gender: {gender}, Age: {age}")

    # Draw bounding boxes around detected faces
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Calculate and display FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Webcam Face Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
