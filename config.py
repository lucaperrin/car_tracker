# Configuration settings for the IP Camera Car Counter application

# Video source settings
VIDEO_SOURCE = "ip_camera"  # Options: "ip_camera" or "demo_video"

# IP Camera settings

CAMERA_URL = "http://192.168.1.200/mjpeg/stream"  # Replace with your IP camera URL
CAMERA_USERNAME = ""  # If authentication is required
CAMERA_PASSWORD = ""  # If authentication is required

# Demo video settings
DEMO_VIDEO_PATH = "demo/demo.mov"  # Path to demo video file

# Processing settings
FRAME_WIDTH = 640  # Width to resize frames to
FRAME_HEIGHT = 480  # Height to resize frames to
PROCESS_EVERY_N_FRAMES = 1  # Process every frame (was 3)

# YOLO settings
YOLO_MODEL = "yolov8n.pt"  # Model to use (n=nano, s=small, m=medium, l=large, x=xlarge)
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detection
CLASSES_TO_DETECT = [2, 3, 5, 7, 1]  # Class IDs for vehicles (car, motorcycle, bus, truck) and cyclist (bicycle)

# Tracking settings
MAX_DISAPPEARED = 15  # Maximum number of frames an object can disappear
MAX_DISTANCE = 150  # Maximum distance between centroids to consider it the same object
DIRECTION_THRESHOLD = 5  # Minimum horizontal movement to determine direction

# Web interface settings
WEB_PORT = 5000  # Port for the web interface
DEBUG_MODE = True  # Enable debug mode