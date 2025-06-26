# Player Multi-View Tracking and ID Assignment

This project performs player detection, tracking, and consistent ID assignment across two video views (e.g., front and side cameras). It uses YOLOv11 for object detection, DeepSORT for tracking, and cosine similarity on deep features to match players across different angles.

---

## 📁 Project Structure

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

## 🚀 How It Works

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

## 🧾 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## ✅ Features

* Player-only global ID mapping
* Robust to occlusions and short-term missing detections
* Support for YOLO custom class names
* Fast cosine-based feature comparison
* Clear, readable final videos

---

## 🔍 Class Mapping Used

```python
CLASS_NAMES = {
    0: "Ball",
    1: "Goalkeeper",
    2: "Player",
    3: "Referee"
}
```

---

## 📌 Notes

* This project focuses on visual consistency, not physical tracking (e.g., world coordinates)
* I can improve matching using ball or cameraman as spatial reference but i really don't work in that parts so i don't know how to implement in code
* Currently IDs are consistent only for players (class 2)

---
