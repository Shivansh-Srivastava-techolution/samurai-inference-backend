def get_object_polygons(image):
    return []

def find_bbox_in_first_n_frames(cap, max_frames=5):
    for i in range(max_frames):
        ret, frame = cap.read()
    
        if not ret:
            print(f"Failed to read frame: {i + 1}")
            return None

        bboxes = get_object_polygons(frame)

        if len(bboxes) > 0:
            print(f"Found bounding boxes in frame: {i + 1}")
            return bboxes  # Or return frame, bboxes if you need the frame too

    print(f"No bounding boxes found in first {max_frames} frames")
    return None
