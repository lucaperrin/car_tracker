import cv2
import requests
import numpy as np
import time
from detector import VehicleDetector
from tracker import VehicleTracker
from config import FRAME_WIDTH, FRAME_HEIGHT, PROCESS_EVERY_N_FRAMES, CAMERA_URL

def test_with_esp32_stream():
    detector = VehicleDetector()
    tracker = VehicleTracker()
    tracker.set_counting_line(FRAME_WIDTH)

    stream = requests.get(CAMERA_URL, stream=True)
    bytes_data = b''
    frame_count = 0

    cv2.namedWindow('ESP32-CAM Car Counter', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ESP32-CAM Car Counter', FRAME_WIDTH, FRAME_HEIGHT)
    print("Press 'q' to quit")

    try:
        for chunk in stream.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                frame = cv2.flip(frame, -1)  # Flip vertically and horizontally
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                frame_count += 1
                if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                    detections = detector.detect(frame)
                    tracker.update(detections)
                    frame = detector.draw_detections(frame, detections)
                    frame = tracker.draw_tracking(frame)
                cv2.imshow('ESP32-CAM Car Counter', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit by user")
                    break
                time.sleep(0.01)
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        cv2.destroyAllWindows()
        print(f"Final count: {tracker.left_to_right_count} cars moving left to right")

if __name__ == "__main__":
    print("** RUNNING ESP32-CAM STREAM MODE **")
    test_with_esp32_stream()
