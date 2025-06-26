# utils/feature_utils.py
import json
import cv2
import torch
import numpy as np
import torchreid
from torchvision import transforms

# Preprocessing for input crops
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

# Load ReID model once
reid_model = torchreid.models.build_model(
    name='osnet_x1_0', num_classes=1000, pretrained=True
)
reid_model.eval()
reid_model.cuda()

def extract_features(detections_json_path, video_path, save_path):
    with open(detections_json_path) as f:
        detections = json.load(f)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    all_features = {}

    while True:
        ret, frame = cap.read()
        if not ret or frame_id >= len(detections):
            break

        dets = detections[frame_id]
        for i, det in enumerate(dets):
            if len(det) == 7:
                x1, y1, x2, y2, track_id, _, _ = det
            else:
                continue  # skip if format is not as expected
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                continue
            inp = transform(crop).unsqueeze(0).cuda()
            with torch.no_grad():
                feat = reid_model(inp).cpu().numpy().flatten()
            local_id = f"{'front' if 'front' in video_path else 'side'}_{int(track_id)}"
            all_features[local_id] = feat.tolist()

        frame_id += 1

    cap.release()
    np.save(save_path, all_features)
    print(f"Saved features to {save_path}")
