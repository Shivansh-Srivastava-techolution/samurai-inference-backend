import os
import uuid
import time
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.samurai_ai.main import inference
from app.utils.route_helper import save_temp_video


router = APIRouter()

@router.post("/samurai_inference")
async def process_video(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    try:
        # Save uploaded video temporarily
        video_path = save_temp_video(file)

        samurai_time_taken, cnn_time, gdino_time, logic_result, model_result = inference(video_path)

        response = {
            "inference": {
                "model_inference": model_result,
                "logic_inference": logic_result,
                "metadata": {
                    "filename": file.filename,
                    "file_size_kb": round(os.path.getsize(video_path) / 1024, 2),
                    "video_id": str(uuid.uuid4()),
                    "gdino_inference_time": gdino_time,
                    "cnn_inference_time": cnn_time,
                    "samurai_inference_time": samurai_time_taken
            }
            }
        }
        return JSONResponse(content=response)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)