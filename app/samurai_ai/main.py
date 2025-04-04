import os
import cv2
import time
import torch
from app.core.Config import SAMURAI_WEIGHT_PATH ,CNN_WEIGHT_PATH, CLASS_MAP
from app.utils.cnn_helper import compute_features
from app.utils.bbox_helper import find_bbox_in_first_n_frames, bbox_to_xywh
from app.samurai_ai.samurai_inference import process_video
from app.samurai_ai.cnn_model import CNN1DModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_map = CLASS_MAP

# Create and load the CNN1DModel
cnn_model = CNN1DModel(num_features=9, num_classes=len(class_map))
cnn_model.load_state_dict(torch.load(CNN_WEIGHT_PATH, map_location=device))
cnn_model.to(device)

print("CNN Model Loaded:")
print(cnn_model)
print(device)
print(class_map)

def inference(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bbox_time_start = time.perf_counter()
    bboxes = find_bbox_in_first_n_frames(cap)
    bbox_time_taken = time.perf_counter() - bbox_time_start
    cap.release()

    # Convert bounding boxes to x, y, w, h format
    samurai_bboxes = bbox_to_xywh(bboxes)

    # Track bounding boxes across frames
    vidname = os.path.basename(video_path)
    sam_save_path = os.path.join("sam2_results", f"track_{vidname}")
    os.makedirs("sam2_results", exist_ok=True)
    logic_inference, bbox_sequence = process_video(video_path, samurai_bboxes, model_path=SAMURAI_WEIGHT_PATH, 
                save_video=False, output_path=sam_save_path)

    # Compute features for model
    features = compute_features(bbox_sequence, frame_width, frame_height)

    # Convert to (1, seq_len, 9) for CNN
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    # Inference for classification
    cnn_model.eval()
    with torch.no_grad():
        outputs = cnn_model(input_tensor)  # => shape [1, 2]
        predicted_idx = torch.argmax(outputs, dim=1).item()

    model_inference = class_map.get(predicted_idx, "unknown")

    return bbox_time_taken, logic_inference, model_inference, 