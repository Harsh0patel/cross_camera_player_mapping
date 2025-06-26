# Player Multi-View Tracking and ID Assignment

This project performs player detection, tracking, and consistent ID assignment across two video views (e.g., front and side cameras). It uses YOLO for object detection, DeepSORT for tracking, and cosine similarity on deep features to match players across different angles.

---

## Project Structure

```
.
├── best.pt                    # Trained YOLO model
├── detections/               # JSONs storing detection results
├── features/                 # Extracted feature vectors (npy)
├── matches/                  # Global ID maps
├── output/                   # Final annotated videos
├── utils/
│   ├── draw_utils.py         # Video annotation and rendering
│   ├── feature_utils.py      # Extract deep features from player crops
│   └── matcher.py            # Assign consistent global IDs
├── front.mp4                 # First video (top/front view)
├── side.mp4                  # Second video (side/close view)
```

---

## How It Works

### 1. Detection + Tracking

* YOLO detects players, referees, goalkeepers, and ball.
* DeepSORT tracks players over time, assigning unique track IDs per video.

### 2. Feature Extraction

* Crop player regions
* Use TorchreID + pretrained OSNet model to extract a deep feature vector for each player

### 3. Global ID Matching

* Use cosine similarity between player features in both videos
* Match players with high similarity and assign global consistent IDs
* Players with no match are assigned local IDs (fallback: 10000+track\_id)

### 4. Rendering Output

* Annotated videos are rendered with:

  * Stable smoothed boxes
  * Class-specific colors
  * Consistent IDs across views
  * Minimum lifespan filter (3+ frames) for cleaner results

---

## Techniques We Tried

* YOLOv11 for robust detection (ball, player, referee, goalkeeper)
* DeepSORT for local tracking with stable track IDs
* OSNet from TorchreID for person ReID feature extraction
* Cosine similarity to match identities between views
* Class filtering to only match `player` class
* Smoothed bounding boxes using a frame buffer
* Local fallback IDs for unmatched players (10000+track\_id)
* Confidence and size thresholds to remove noise
* Frame-based consistency to skip 1-frame flickers

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

```python
# Step 1: Detect and track with YOLO and DeepSORT
# Already done in your preprocessed video and JSONs

# Step 2: Extract features (only class 2 - players)
extract_features('detections/front.json', 'front.mp4', 'features/front.npy')
extract_features('detections/side.json', 'side.mp4', 'features/side.npy')

# Step 3: Match across views
match_tracks('features/front.npy', 'features/side.npy', 'matches/global_id_map.json')

# Step 4: Render videos
render_video('front.mp4', 'detections/front.json', global_id_map, 'output/front_output.mp4')
render_video('side.mp4', 'detections/side.json', global_id_map, 'output/side_output.mp4')
```

---

## Features

* Player-only global ID mapping
* Robust to occlusions and short-term missing detections
* Support for YOLO custom class names
* Fast cosine-based feature comparison
* Clear, readable final videos

---

## Class Mapping Used

```python
CLASS_NAMES = {
    0: "Ball",
    1: "Goalkeeper",
    2: "Player",
    3: "Referee"
}
```

---

## Notes

* This project focuses on visual consistency, not physical tracking (e.g., world coordinates)
* i can later improve matching using ball or cameraman as spatial reference
* Currently IDs are consistent only for players (class 2)
* and there are some misclassified errors

---
