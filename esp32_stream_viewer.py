import cv2
import requests
import numpy as np

# Replace with your ESP32-CAM stream URL (usually ends with /stream)
ESP32_STREAM_URL = "http://192.168.1.200/mjpeg/stream"

def main():
    stream = requests.get(ESP32_STREAM_URL, stream=True)
    bytes_data = b''
    print(f"Connecting to ESP32-CAM stream at {ESP32_STREAM_URL} ...")
    for chunk in stream.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.flip(frame, -1)  # Flip vertically and horizontally
                cv2.imshow('ESP32-CAM Stream', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
