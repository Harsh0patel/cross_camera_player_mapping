# utils/tracker.py
import os
import json
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize DeepSort once
tracker = DeepSort(max_age=30)

def run_tracking(video_path, model_path, output_json_path):
    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)
    frame_id = 0
    tracked_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]  # YOLOv8 single-frame inference
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))  # [x, y, w, h]

        tracks = tracker.update_tracks(detections, frame=frame)

        frame_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            x1, y1, x2, y2 = ltrb
            frame_tracks.append([x1, y1, x2, y2, track_id, track.det_class, track.det_conf])

        tracked_data.append(frame_tracks)
        frame_id += 1

    cap.release()

    # Save to JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(tracked_data, f)

    print(f"Tracking data saved to {output_json_path}")