import numpy as np

def normalize_bbox(bbox, frame_width, frame_height):
    x, y, w, h = bbox
    return [
        x / frame_width,
        y / frame_height,
        w / frame_width,
        h / frame_height
    ]

def compute_features(bbox_sequence, frame_width, frame_height):
    features = []
    for i in range(len(bbox_sequence)):
        x, y, w, h = bbox_sequence[i]

        # Normalized bounding box
        norm_bbox = normalize_bbox([x, y, w, h], frame_width, frame_height)

        # Compute displacement, velocity, aspect ratio, and area
        if i > 0:
            prev_x, prev_y, prev_w, prev_h = bbox_sequence[i - 1]
            displacement = [x - prev_x, y - prev_y]
            velocity = np.sqrt(displacement[0]**2 + displacement[1]**2)
        else:
            displacement = [0, 0]
            velocity = 0

        aspect_ratio = w / h if h != 0 else 0
        area = (w * h) / (frame_width * frame_height)

        # Convert all values to Python native types
        feature = norm_bbox + [float(d) for d in displacement] + [float(velocity), float(aspect_ratio), float(area)]
        features.append(feature)
    return features