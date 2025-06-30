import numpy as np
import cv2
from scipy.spatial import distance
from collections import OrderedDict
from config import MAX_DISAPPEARED, MAX_DISTANCE, DIRECTION_THRESHOLD
import os
import time

class VehicleTracker:
    def __init__(self):
        # Initialize the next unique object ID
        self.next_object_id = 0
        
        # Initialize dictionaries to store object info
        self.objects = OrderedDict()  # Maps object_id -> centroid
        self.disappeared = OrderedDict()  # Maps object_id -> number of frames disappeared
        self.directions = OrderedDict()  # Maps object_id -> direction (1=right, -1=left, 0=unknown)
        self.previous_positions = OrderedDict()  # Maps object_id -> previous centroid
        self.counted = set()  # Set of object IDs that have been counted
        self.direction_history = OrderedDict()  # Maps object_id -> list of recent directions
        self.last_counted_time = OrderedDict()  # Maps object_id -> last counted frame number
        self.classes = OrderedDict()  # Maps object_id -> vehicle class
        
        # Initialize counts dictionary for each class
        self.counts = {
            'car': 0,
            'truck': 0,
            'motorcycle': 0,
            'bus': 0,
            'person': 0,
            'bicycle': 0  # Added cyclist
        }
        
        # Counters for both directions
        self.left_to_right_count = 0
        self.right_to_left_count = 0
        
        # Counters by class for both directions
        self.left_to_right_by_class = {
            'car': 0,
            'truck': 0,
            'motorcycle': 0,
            'bus': 0,
            'person': 0,
            'bicycle': 0  # Added cyclist
        }
        self.right_to_left_by_class = {
            'car': 0,
            'truck': 0,
            'motorcycle': 0,
            'bus': 0,
            'person': 0,
            'bicycle': 0  # Added cyclist
        }
        
        # Store parameters
        self.max_disappeared = MAX_DISAPPEARED
        self.max_distance = MAX_DISTANCE
        self.direction_threshold = DIRECTION_THRESHOLD
        
        # Define counting line (vertical line instead of horizontal)
        self.counting_line_x = None
        
        # New parameters for improved counting
        self.direction_history_length = 5  # Number of frames to confirm direction
        self.counting_cooldown = 30  # Frames to wait before counting same object again
        
        # Frame counter
        self.frame_count = 0
        
        # Added for _check_counting_line method
        self.counting_cooldown = 10  # Assuming a default value, actual implementation needed
        self.min_counting_distance = 50  # Assuming a default value, actual implementation needed
        
        # Reset the counts file
        self.reset_counts_file()
    
    def reset_counts_file(self):
        """Reset the counts file with headers"""
        with open('vehicle_counts.txt', 'w') as f:
            f.write("Vehicle Counting Results\n")
            f.write("=======================\n\n")
            f.write("Left to Right Counts:\n")
            f.write("-------------------\n")
            for class_name in self.left_to_right_by_class.keys():
                f.write(f"{class_name}: 0\n")
            f.write("\nRight to Left Counts:\n")
            f.write("-------------------\n")
            for class_name in self.right_to_left_by_class.keys():
                f.write(f"{class_name}: 0\n")
            f.write("\nTotal Counts:\n")
            f.write("------------\n")
            f.write("Left to Right Total: 0\n")
            f.write("Right to Left Total: 0\n")

    def save_counts_to_file(self):
        """Save current counts to file"""
        with open('vehicle_counts.txt', 'w') as f:
            f.write("Vehicle Counting Results\n")
            f.write("=======================\n\n")
            
            # Left to Right counts
            f.write("Left to Right Counts:\n")
            f.write("-------------------\n")
            for class_name, count in self.left_to_right_by_class.items():
                f.write(f"{class_name}: {count}\n")
            
            # Right to Left counts
            f.write("\nRight to Left Counts:\n")
            f.write("-------------------\n")
            for class_name, count in self.right_to_left_by_class.items():
                f.write(f"{class_name}: {count}\n")
            
            # Total counts
            f.write("\nTotal Counts:\n")
            f.write("------------\n")
            f.write(f"Left to Right Total: {self.left_to_right_count}\n")
            f.write(f"Right to Left Total: {self.right_to_left_count}\n")

    def set_counting_line(self, frame_width):
        """Set the counting line position"""
        # Set a vertical line at the middle of the frame width
        self.counting_line_x = frame_width // 2  # Changed from frame_width // 3 to frame_width // 2
    
    def register(self, centroid, class_id):
        """Register a new object with the tracker"""
        # Generate a new ID
        self.next_object_id += 1
        
        # Store the centroid and class
        self.objects[self.next_object_id] = centroid
        self.previous_positions[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.directions[self.next_object_id] = None
        self.classes[self.next_object_id] = class_id
        
        # Initialize counters for this class if not exists
        if class_id not in self.counts:
            self.counts[class_id] = 0
    
    def deregister(self, object_id):
        """Deregister an object with the given ID"""
        # Delete the object from our dictionaries
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.directions[object_id]
        del self.previous_positions[object_id]
        if object_id in self.direction_history:
            del self.direction_history[object_id]
        if object_id in self.classes:
            del self.classes[object_id]
        if hasattr(self, 'vertical_zone'):
            self.vertical_zone = None
    
    def update(self, detections, frame=None):
        """Update the tracked objects with new detections"""
        # Increment frame counter
        self.frame_count += 1
        
        # Extract centroids and classes from detections
        centroids = np.array([d['centroid'] for d in detections]) if detections else np.empty((0, 2))
        classes = [d['class'] for d in detections] if detections else []

        # Map object_id to detection for saving crops
        detection_map = {}
        for d in detections:
            detection_map[d['centroid']] = d

        # If no tracked objects and no detections, do nothing
        if len(self.objects) == 0 and len(centroids) == 0:
            return self.objects

        # If we have no tracked objects, register all detections
        if len(self.objects) == 0:
            for centroid, class_id in zip(centroids, classes):
                self.register(centroid, class_id)
        # If there are no detections, mark all objects as disappeared
        elif len(centroids) == 0:
            for object_id in list(self.objects.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        # Otherwise, match existing objects with new detections
        else:
            # Get the IDs and centroids of current objects
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Apply motion prediction for existing objects
            predicted_centroids = []
            for object_id in object_ids:
                current = self.objects[object_id]
                previous = self.previous_positions[object_id]
                
                # Calculate velocity vector
                dx = current[0] - previous[0]
                dy = current[1] - previous[1]
                
                # Predict next position based on current velocity
                predicted_x = current[0] + dx
                predicted_y = current[1] + dy
                
                predicted_centroids.append((predicted_x, predicted_y))
            
            # Compute the distance between each pair of predicted centroids and detection centroids
            D = distance.cdist(np.array(predicted_centroids), centroids)
            
            # Find the smallest value in each row and the index of the column
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            # Keep track of which rows and columns we've already examined
            used_rows = set()
            used_cols = set()
            
            # Loop over the combinations of (row, column) index tuples
            for (row, col) in zip(rows, cols):
                # If this row or column has already been used, or the distance is too large, skip
                if row in used_rows or col in used_cols or D[row, col] > self.max_distance:
                    continue
                
                # Otherwise, get the object ID for the current row
                object_id = object_ids[row]
                
                # Update the centroid and reset disappeared counter
                prev_centroid = self.objects[object_id]
                self.previous_positions[object_id] = prev_centroid
                self.objects[object_id] = centroids[col]
                self.disappeared[object_id] = 0
                
                # Update direction
                self._update_direction(object_id)
                
                # Check if object crossed the counting line, pass detection
                detection = None
                for d in detections:
                    if np.allclose(d['centroid'], centroids[col]):
                        detection = d
                        break
                self._check_counting_line(object_id, frame, detection)
                
                # Mark the row and column as used
                used_rows.add(row)
                used_cols.add(col)
            
            # Compute the unused rows and columns
            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)
            
            # If we have more objects than detections, mark objects as disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    # Deregister the object if it's been gone for too long
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            
            # Otherwise, register each new detection
            else:
                for col in unused_cols:
                    self.register(centroids[col], classes[col])
        
        # Return the updated objects
        return self.objects
    
    def _update_direction(self, object_id):
        """Update the direction of an object based on its movement"""
        current = self.objects[object_id]
        previous = self.previous_positions[object_id]
        
        # Calculate movement
        dx = current[0] - previous[0]
        
        # Update direction based on movement
        if dx > 0:
            self.directions[object_id] = 'right'
        elif dx < 0:
            self.directions[object_id] = 'left'
    
    def _check_counting_line(self, object_id, frame=None, detection=None):
        """Check if an object has crossed the counting line and update counts"""
        # Get current and previous positions
        current = self.objects[object_id]
        previous = self.previous_positions[object_id]

        # Get class name
        class_id = self.classes[object_id]
        class_name = self._get_class_name(class_id)

        # Only count if the object moved enough horizontally (not parked)
        if abs(current[0] - previous[0]) < self.direction_threshold:
            return  # Ignore objects that haven't moved enough

        # Check if the object crossed the line
        if (previous[0] < self.counting_line_x and current[0] >= self.counting_line_x):
            if self.directions[object_id] == 'right':
                self.left_to_right_count += 1
                self.left_to_right_by_class[class_name] += 1
                print(f"Counted {class_name} moving right. Total: {self.left_to_right_by_class[class_name]}")
                self.save_counts_to_file()
                # Save cropped image
                if frame is not None and detection is not None:
                    x1, y1, x2, y2 = detection['bbox']
                    crop = frame[y1:y2, x1:x2]
                    os.makedirs('crossings', exist_ok=True)
                    filename = f"crossings/{class_name}_right_{object_id}_{int(time.time())}.jpg"
                    cv2.imwrite(filename, crop)
        elif (previous[0] > self.counting_line_x and current[0] <= self.counting_line_x):
            if self.directions[object_id] == 'left':
                self.right_to_left_count += 1
                self.right_to_left_by_class[class_name] += 1
                print(f"Counted {class_name} moving left. Total: {self.right_to_left_by_class[class_name]}")
                self.save_counts_to_file()
                # Save cropped image
                if frame is not None and detection is not None:
                    x1, y1, x2, y2 = detection['bbox']
                    crop = frame[y1:y2, x1:x2]
                    os.makedirs('crossings', exist_ok=True)
                    filename = f"crossings/{class_name}_left_{object_id}_{int(time.time())}.jpg"
                    cv2.imwrite(filename, crop)
    
    def _get_class_name(self, class_id):
        """Convert class ID to name"""
        class_map = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            0: 'person',
            1: 'bicycle'  # Added cyclist
        }
        return class_map.get(class_id, str(class_id))
    
    def draw_tracking(self, frame):
        """Draw only the counting line, green box, ID, and confidence for each object."""
        # Draw counting line (vertical line)
        if self.counting_line_x is not None:
            cv2.line(frame, (self.counting_line_x, 0), (self.counting_line_x, frame.shape[0]), (0, 255, 255), 4)  # Yellow line, thicker
        for object_id, centroid in self.objects.items():
            class_id = self.classes.get(object_id)
            prev_centroid = self.previous_positions.get(object_id, centroid)
            dx = abs(centroid[0] - prev_centroid[0])
            # Only show ID if movement is above threshold
            if dx >= self.direction_threshold:
                text = f"ID {object_id}"
                color = (0, 255, 0)  # Green for ID text
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # The green box and confidence are drawn in detector.draw_detections
        return frame