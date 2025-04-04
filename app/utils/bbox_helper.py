import cv2
import io
import uuid
import requests
from app.core.Config import GDINO_URL, GDINO_PROMPT

def bbox_to_xywh(bbox):
    # Extract x and y values from the points
    x_coords = [p["x"] for p in bbox]
    y_coords = [p["y"] for p in bbox]

    # Calculate bounding box dimensions
    x_min = min(x_coords)
    y_min = min(y_coords)
    width = max(x_coords) - x_min
    height = max(y_coords) - y_min

    return [x_min, y_min, width, height]

def get_object_polygons(frame):

    success, encoded_image = cv2.imencode('.png', frame)
    if not success:
        raise Exception ("Failed to encode image")
    
    image_bytes = io.BytesIO(encoded_image.tobytes())
    # Request setup
    url = GDINO_URL
    payload = {'prompt': GDINO_PROMPT}

    files = [
        ('image', (f'{uuid.uuid4()}_frame.png', image_bytes, 'image/png'))
    ]

    response = requests.post(url, data=payload, files=files)
    return response.json() 

def find_bbox_in_first_n_frames(cap, max_frames=5):
    for i in range(max_frames):
        ret, frame = cap.read()
    
        if not ret:
            print(f"Failed to read frame: {i + 1}")
            return None

        json_response = get_object_polygons(frame)
        bboxes = json_response["bboxes"]

        if len(bboxes) > 0:
            print(f"Found bounding boxes in frame: {i + 1}")
            return bboxes

    print(f"No bounding boxes found in first {max_frames} frames")
    return None
