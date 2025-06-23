# Minimal Car/Cyclist Counter (Local Testing)

This project is now set up for local testing only, using either an ESP32-CAM stream or a video file. The web interface and related files have been removed.

## How to Use

1. Install dependencies (see `INSTALL.md`).
2. To test with an ESP32 stream, run:
   ```bash
   python test_with_esp32_stream.py
   ```
3. To test with a video file, edit `test_with_video.py` to set your video path, then run:
   ```bash
   python test_with_video.py
   ```

- Results and counts will be printed in the terminal and saved to `vehicle_counts.txt`.
- For configuration, edit `config.py`.

## Requirements
See `requirements.txt` for the minimal set of required Python packages.