import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from utils.helper import save_temp_video

app = FastAPI()

@app.get("/")
def get():
    return {"message": "Samurai Inference Backend!"}

@app.post("/samurai-inference")
async def process_video(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    try:
        # Save uploaded video temporarily
        video_path = save_temp_video(file)

        # Run your custom inferences
        model_result = "grab"
        logic_result = "invalid"

        response = {
            "inference": {
                "model_inference": model_result,
                "logic_inference": logic_result,
                "metadata": {
                    "filename": file.filename,
                    "file_size_kb": round(os.path.getsize(video_path) / 1024, 2),
                    "video_id": str(uuid.uuid4())
            }
            }
        }
        return JSONResponse(content=response)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
