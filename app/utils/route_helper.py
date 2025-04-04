import shutil
import uuid
import os

def save_temp_video(file) -> str:
    temp_filename = f"temp_{uuid.uuid4()}.mp4"
    temp_path = os.path.join("/tmp", temp_filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_path
