import cv2
import time
import numpy as np
import psutil  # Added for CPU usage
from detector import VehicleDetector
from tracker import VehicleTracker
from config import FRAME_WIDTH, FRAME_HEIGHT, PROCESS_EVERY_N_FRAMES, DEMO_VIDEO_PATH

def test_with_video(video_path):
    """Test the car counting system with a video file"""
    # Initialize components
    detector = VehicleDetector()
    tracker = VehicleTracker()
    
    # Set the counting line position
    tracker.set_counting_line(FRAME_WIDTH)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    frame_count = 0
    
    # Create a window
    cv2.namedWindow('Car Counter Test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Car Counter Test', FRAME_WIDTH, FRAME_HEIGHT)
    
    print("Press 'q' to quit")
    start_wall_time = time.time()  # Real clock
    start_virtual_time = None      # Video clock
    try:
        while True:
            # Read a frame
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break
            
            # Get video timestamp (if available)
            video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # seconds
            if start_virtual_time is None:
                start_virtual_time = video_time
            
            # Resize frame
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            
            # Process every Nth frame
            frame_count += 1
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                # Detect vehicles
                detections = detector.detect(frame)
                
                # Update tracker with detections
                tracker.update(detections)
                
                # Draw detections and tracking info
                frame = detector.draw_detections(frame, detections)
                frame = tracker.draw_tracking(frame)
            
            # Display the frame
            cv2.imshow('Car Counter Test', frame)
            
            # Print CPU usage and time info
            wall_elapsed = time.time() - start_wall_time
            video_elapsed = video_time - start_virtual_time if start_virtual_time is not None else 0
            print(f"CPU usage: {psutil.cpu_percent(interval=None)}% | Wall: {wall_elapsed:.2f}s | Video: {video_elapsed:.2f}s | Delay: {wall_elapsed - video_elapsed:.2f}s")
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit by user")
                break
            
            # Limit processing rate
            time.sleep(0.01)  # Increased from 0.01 to slow down playback
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print(f"Final count: {tracker.left_to_right_count} cars moving left to right")

if __name__ == "__main__":
    # Use DEMO_VIDEO_PATH from config.py
    video_path = DEMO_VIDEO_PATH
    print("** RUNNING DEMO VIDEO MODE **")
    test_with_video(video_path)