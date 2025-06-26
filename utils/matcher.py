# utils/matcher.py
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

def match_tracks(front_feat_path, side_feat_path, save_path, top_k=1):
    front_feats = np.load(front_feat_path, allow_pickle=True).item()
    side_feats = np.load(side_feat_path, allow_pickle=True).item()

    id_map = {}
    global_id = 0

    for f_key, f_feat in front_feats.items():
        similarities = {}
        for s_key, s_feat in side_feats.items():
            sim = cosine_similarity([f_feat], [s_feat])[0][0]
            similarities[s_key] = sim

        # Sort by similarity and take the best matches
        best_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

        for s_key, sim_score in best_matches:
            if f_key not in id_map and s_key not in id_map:
                id_map[f_key] = global_id
                id_map[s_key] = global_id
                global_id += 1

    # Filter to only keep entries where local_id comes from 'front_' or 'side_' (i.e., player)
    filtered_map = {
        k: v for k, v in id_map.items()
        if 'front_' in k or 'side_' in k
    }

    with open(save_path, "w") as f:
        json.dump(filtered_map, f)

    print(f"Saved global ID map to {save_path}")
