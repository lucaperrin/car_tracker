# Minimal Installation and Usage Guide for Local Testing

## Prerequisites

- Python 3.8 or higher
- Pip package manager
- ESP32-CAM or video file for testing

## Clone the Repository (on Ubuntu)

1. Open a terminal and run:

   ```bash
   git clone https://github.com/lucaperrin/car_counter.git
   cd car_counter/processing
   ```

## Installation Steps

1. (Recommended) Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Upgrade pip:

   ```bash
   pip install --upgrade pip
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

- To test with ESP32 stream:

  ```bash
  python test_with_esp32_stream.py
  ```

- To test with a video file:

  Edit `test_with_video.py` to set your video path, then run:

  ```bash
  python test_with_video.py
  ```

## Notes

- No web interface is included in this minimal setup.
- All results and counts will be shown in the terminal or written to `vehicle_counts.txt`.