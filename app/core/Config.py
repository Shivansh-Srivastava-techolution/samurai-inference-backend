import os
import json

FRAME_SKIP = 3
CNN_WEIGHT_PATH = "app/weights/cnn1d_model.pth"
SAMURAI_WEIGHT_PATH = "app/weights/sam2.1_hiera_large.pt"
CLASS_MAP_PATH = "app/weights/cls_map.json"
SAVE_SAM2_VIDEO = True

if os.path.exists(CLASS_MAP_PATH):
    with open(CLASS_MAP_PATH, "r") as f:
        CLASS_MAP = json.load(f) 
else:
    CLASS_MAP = {}

GDINO_URL = "http://35.232.107.25:8501/gdino_infer"
GDINO_PROMPT = 'medical paper, box'