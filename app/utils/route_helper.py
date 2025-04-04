import shutil
import uuid
import os
import cv2

def save_temp_video(file) -> str:
    temp_filename = f"temp_{uuid.uuid4()}.mp4"
    temp_path = os.path.join("/tmp", temp_filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_path

def preprocess_saved_video(saved_video_path: str, frame_skip: int = 1, resolution_factor: float = 1) -> str:
    """
    Preprocess a saved video: resize frames and/or skip frames, and save as a new video.

    Args:
        saved_video_path (str): Path to the original saved video file.
        frame_skip (int): Number of frames to skip (e.g., 5 means use every 5th frame).
        resize_to (tuple): Resize dimensions (width, height). Pass None to keep original size.

    Returns:
        str: Path to the preprocessed output video file.
    """
    if not os.path.isfile(saved_video_path):
        raise FileNotFoundError(f"Video not found: {saved_video_path}")

    cap = cv2.VideoCapture(saved_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {saved_video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Original data:", original_fps, orig_w, orig_h)

    # Determine output size
    out_w, out_h = resolution_factor * orig_w, resolution_factor * orig_h
    print("Preprocessed data:", original_fps, out_w, out_h)

    # Output file path
    output_path = saved_video_path.replace(".mp4", "_processed.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, original_fps, (out_w, out_h))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_skip == 0:
            if resolution_factor != 1:
                frame = cv2.resize(frame, (out_w, out_h))
            out_writer.write(frame)
        idx += 1

    cap.release()
    out_writer.release()

    return output_path

