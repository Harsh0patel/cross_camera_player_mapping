# utils/draw_utils.py
import cv2
from collections import defaultdict, deque

# Correct class definitions from model
CLASS_NAMES = {
    0: "Ball",
    1: "Goalkeeper",
    2: "Player",
    3: "Referee"
}

CLASS_COLORS = {
    0: (0, 165, 255),     # Ball - orange
    1: (255, 0, 0),       # Goalkeeper - blue
    2: (0, 255, 0),       # Player - green
    3: (0, 255, 255)      # Referee - yellow
}

# Track smoothing buffer
track_history = defaultdict(lambda: deque(maxlen=5))
track_lifetime = defaultdict(int)

def draw_boxes(frame, detections, global_id_map, frame_id, view='front'):
    for det in detections:
        if len(det) == 7:
            x1, y1, x2, y2, track_id, cls, conf = det
        else:
            continue

        if conf is None or float(conf) < 0.5:
            continue

        if (x2 - x1) * (y2 - y1) < 500:
            continue

        # Smooth position
        track_id = int(track_id)
        box = [int(x1), int(y1), int(x2), int(y2)]
        track_history[track_id].append(box)
        avg_box = list(map(lambda vals: int(sum(vals) / len(vals)), zip(*track_history[track_id])))

        # Count track appearance
        track_lifetime[track_id] += 1
        if track_lifetime[track_id] < 3:
            continue  # only show if track exists for >=3 frames

        local_id = f"{view}_{track_id}"

        if local_id in global_id_map:
            global_id = global_id_map[local_id]
        else:
            global_id = 100 + track_id  # fallback stable local ID

        color = CLASS_COLORS.get(int(cls), (255, 255, 255))
        class_name = CLASS_NAMES.get(int(cls), "Unknown")
        label = f"{class_name} | ID: {global_id}"

        x1_s, y1_s, x2_s, y2_s = avg_box
        cv2.rectangle(frame, (x1_s, y1_s), (x2_s, y2_s), color, 2)
        cv2.putText(frame, label, (x1_s, y1_s - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame
