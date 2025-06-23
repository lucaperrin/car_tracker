import cv2
import torch
import numpy as np
from ultralytics import YOLO
from config import YOLO_MODEL, CONFIDENCE_THRESHOLD, CLASSES_TO_DETECT

class VehicleDetector:
    def __init__(self):
        self.model_name = YOLO_MODEL
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.classes_to_detect = CLASSES_TO_DETECT
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()
        
    def load_model(self):
        """Load the YOLO model"""
        try:
            print(f"Loading YOLO model {self.model_name} on {self.device}...")
            self.model = YOLO(self.model_name)
            print(f"YOLO model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading YOLO model: {str(e)}")
            return False
    
    def detect(self, frame):
        """Detect vehicles in the frame"""
        if self.model is None:
            print("Model not loaded")
            return []
        
        try:
            # Apply slight Gaussian blur to reduce noise and help with detection
            blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
            
            # Run inference
            results = self.model(blurred_frame, verbose=False)
            
            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class, confidence and bounding box
                    cls = int(box.cls.item())
                    conf = box.conf.item()
                    
                    # Filter by class and confidence
                    if cls in self.classes_to_detect and conf >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        # Calculate centroid
                        centroid_x = (x1 + x2) // 2
                        centroid_y = (y1 + y2) // 2
                        
                        detections.append({
                            'class': cls,
                            'confidence': conf,
                            'bbox': (x1, y1, x2, y2),
                            'centroid': (centroid_x, centroid_y)
                        })
            
            return detections
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return []
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on the frame"""
        class_names = {
            2: 'Car',
            3: 'Motorcycle',
            5: 'Bus',
            7: 'Truck',
            0: 'Person'
        }
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cls = detection['class']
            conf = detection['confidence']
            centroid = detection['centroid']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw centroid
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
            
            # Draw label with class and confidence
            label = f"{class_names.get(cls, 'Vehicle')}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame