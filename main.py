import cv2
import numpy as np
import os
import hmac
import hashlib
import requests
import json
import base64
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from io import BytesIO

# Supabase Admin (service role) to update rows server-side
from supabase import create_client, Client

# The core API client for Gemini
API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
IMAGE_GEN_MODEL = "imagen-3.0-generate-002"

app = FastAPI()

# CORS (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#--------- Environment / Config ----------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
QR_HMAC_SECRET = os.environ.get("QR_HMAC_SECRET", "dev-secret")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") 

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Geofence and time window (customize these)
GEOFENCE_CENTER = (12.9716, 77.5946) # (lat, lon)
GEOFENCE_RADIUS_M = 150 # meters
WINDOW_START = datetime(2025, 8, 22, 8, 0, 0, tzinfo=timezone.utc)
WINDOW_END = datetime(2025, 8, 22, 18, 0, 0, tzinfo=timezone.utc)


#--------- Helper Functions ----------
def haversine_m(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    """Calculates the distance between two points on Earth using Haversine formula."""
    import math
    R = 6371000.0
    dLat = math.radians(b_lat - a_lat)
    dLon = math.radians(b_lon - a_lon)
    lat1 = math.radians(a_lat)
    lat2 = math.radians(b_lat)
    h = math.sin(dLat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dLon/2)**2
    return 2 * R * math.asin(math.sqrt(h))

def check_geofence(lat: float, lon: float):
    """Checks if the provided coordinates are within the defined geofence."""
    dist = haversine_m(lat, lon, GEOFENCE_CENTER[0], GEOFENCE_CENTER[1])
    if dist <= GEOFENCE_RADIUS_M:
        return True, None
    return False, "Out of allowed zone"

def check_time_window(now: datetime):
    """Checks if the current time is within the allowed time window."""
    if WINDOW_START <= now <= WINDOW_END:
        return True, None
    return False, "Outside allowed time"

def verify_qr_hmac(qr_value: str):
    """Verifies the HMAC signature of a QR code payload."""
    if not qr_value:
        return False, "Missing QR"
    try:
        parts = qr_value.split(".")
        if len(parts) != 2:
            return False, "Invalid QR format"
        payload, sighex = parts[0], parts[1]
        mac = hmac.new(QR_HMAC_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(mac, sighex):
            return False, "QR signature invalid"
        return True, None
    except Exception:
        return False, "QR verification error"

def load_image_from_url(url: str):
    """Fetches an image from a URL and returns it as base64 data."""
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    # Convert image to bytes and then to base64
    img = Image.open(BytesIO(r.content)).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def analyze_image_with_gemini(image_url: str) -> Dict[str, Any]:
    """
    Analyzes an image using the Gemini API.
    It checks for a billboard, reads text, and checks for sensitive content.
    """
    if not GOOGLE_API_KEY:
        return {"status": "error", "message": "API key not found."}

    # Fetch and encode the image
    try:
        image_data = load_image_from_url(image_url)
    except Exception as e:
        return {"status": "error", "message": f"Could not load image from URL: {e}"}

    # Create a structured prompt for Gemini
    prompt_text = (
        "Analyze this image. "
        "1. Is there a clear outdoor billboard or hoarding present? (Answer 'yes' or 'no'). "
        "2. Transcribe any text visible on the billboard. "
        "3. Is the content sensitive, violent, or sexually explicit? (Answer 'yes' or 'no'). "
        "Provide your response as a JSON object with the keys 'is_billboard', 'detected_text', and 'is_sensitive'."
    )
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": image_data
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "is_billboard": {"type": "STRING"},
                    "detected_text": {"type": "STRING"},
                    "is_sensitive": {"type": "STRING"}
                }
            }
        }
    }

    url = f"{API_URL_BASE}{GEMINI_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    
    try:
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        # The response is a JSON string, so we need to parse it
        json_string = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
        result = json.loads(json_string)
        
        return {"status": "success", "data": result}
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return {"status": "error", "message": f"Gemini API request failed: {e}"}
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}. Raw response: {response.text}")
        return {"status": "error", "message": "Failed to parse API response."}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"status": "error", "message": "An unexpected error occurred."}


#--------- Schemas ----------
class AnalyzeResponse(BaseModel):
    status: str # "success" | "violation" | "error"
    message: str
    
#--------- Route ----------
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    image_url: str = Form(...), # public Supabase URL for the stored image
    lat: float = Form(...),
    lon: float = Form(...),
    qr_value: str = Form(""),
    report_id: Optional[str] = Form(None), # Supabase reports.id for server-side update
):
    """Analyzes a reported billboard image for potential violations using AI."""
    violations: List[str] = []

    # 1) Check URL reachability
    try:
        r = requests.head(image_url, timeout=5)
        if r.status_code >= 400:
            violations.append("Image URL not accessible")
    except Exception:
        violations.append("Image URL check failed")

    if violations:
        return {"status": "error", "message": "; ".join(violations)}

    # 2) Geofence & time window
    ok_geo, geo_msg = check_geofence(lat, lon)
    if not ok_geo:
        violations.append(geo_msg)
    
    ok_time, time_msg = check_time_window(datetime.now(timezone.utc))
    if not ok_time:
        violations.append(time_msg)

    # 3) AI-powered analysis with Gemini API
    ai_result = analyze_image_with_gemini(image_url)

    if ai_result["status"] == "error":
        violations.append(f"AI analysis failed: {ai_result['message']}")
    else:
        ai_data = ai_result["data"]
        # Convert to lowercase for reliable comparison
        is_billboard = ai_data.get("is_billboard", "no").lower() == "yes"
        is_sensitive = ai_data.get("is_sensitive", "no").lower() == "yes"
        detected_text = ai_data.get("detected_text", "")

        # Check for billboard presence
        if not is_billboard:
            violations.append("No billboard detected.")
        
        # Check for sensitive content
        if is_sensitive:
            violations.append("Sensitive content detected.")
        
        # Check for license/compliance text (example)
        # We only add this violation if a billboard was detected
        if is_billboard and detected_text and "LIC" not in detected_text.upper():
            violations.append("No license information detected.")

    # 4) QR HMAC verification
    ok_qr, qr_msg = verify_qr_hmac(qr_value)
    if not ok_qr:
        violations.append(qr_msg)

    # 5) Determine final status and update Supabase
    status = "violation" if violations else "success"
    message = "; ".join(violations) if violations else "All checks passed"

    if supabase and report_id:
        try:
            supabase.table("reports").update({"status": status, "message": message}).eq("id", report_id).execute()
        except Exception as e:
            print(f"Failed to update Supabase row: {e}")

    return {"status": status, "message": message}
