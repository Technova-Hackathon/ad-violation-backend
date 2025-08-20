from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

app = FastAPI()

# Allow your Expo app to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ in prod, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze(
    image: UploadFile,
    lat: str = Form(...),
    lon: str = Form(...)
):
    # Save uploaded image
    file_path = os.path.join(UPLOAD_DIR, image.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # TODO: Run ML model here (placeholder for now)
    result = {
        "status": "success",
        "file": image.filename,
        "lat": lat,
        "lon": lon,
        "message": "Image received and stored successfully!"
    }
    return result
