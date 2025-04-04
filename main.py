from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import (
    health_check_router,
    samurai_inference_router,
)

from app.settings.swagger_doc import (
    API_TITLE,
    API_DESCRIPTION,
    API_SUMMARY
)

app = FastAPI(title=API_TITLE,
    description=API_DESCRIPTION,
    summary=API_SUMMARY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_check_router.router)
app.include_router(samurai_inference_router.router)

if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    os.system("echo http://$(curl -s ifconfig.me):8000/samurai_inference")

