from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import hmac
import hashlib
from datetime import datetime, timezone
import requests

import numpy as np
from PIL import Image
from io import BytesIO

#Supabase Admin (service role) to update rows server-side
from supabase import create_client, Client

app = FastAPI()

# CORS (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:19006", "http://127.0.0.1:19006"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#--------- Environment / Config ----------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
QR_HMAC_SECRET = os.environ.get("QR_HMAC_SECRET", "dev-secret") # replace in prod

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Geofence and time window (customize these)
GEOFENCE_CENTER = (12.9716, 77.5946) # (lat, lon)
GEOFENCE_RADIUS_M = 150 # meters
WINDOW_START = datetime(2025, 8, 22, 8, 0, 0, tzinfo=timezone.utc)
WINDOW_END = datetime(2025, 8, 22, 18, 0, 0, tzinfo=timezone.utc)

#--------- Helpers ----------
def haversine_m(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    import math
    R = 6371000.0
    dLat = math.radians(b_lat - a_lat)
    dLon = math.radians(b_lon - a_lon)
    lat1 = math.radians(a_lat)
    lat2 = math.radians(b_lat)
    h = math.sin(dLat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dLon/2)**2
    return 2 * R * math.asin(math.sqrt(h))

def check_geofence(lat: float, lon: float):
    dist = haversine_m(lat, lon, GEOFENCE_CENTER[0], GEOFENCE_CENTER[1])
    if dist <= GEOFENCE_RADIUS_M:
        return True, None
    return False, "Out of allowed zone"

def check_time_window(now: datetime):
    if WINDOW_START <= now <= WINDOW_END:
        return True, None
    return False, "Outside allowed time"

def verify_qr_hmac(qr_value: str):
    """
    Simple signed QR format:
    qr_value = "<payload>.<sighex>"
    where sighex = HMAC-SHA256(secret, payload).hexdigest()

    Client-side, you create:
      payload = "ID12345" (or a JSON/base64 string)
      sighex = hmac.new(QR_HMAC_SECRET, payload, sha256).hexdigest()

    Server checks signature matches.
    """
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
        # Optional: parse payload for exp/nonce and check DB for one-time usage
        return True, None
    except Exception:
        return False, "QR verification error"

#---- NSFW stub (replace with real model later) ----

def nsfw_prob_stub(image_url: str) -> float:
    """
    Stub NSFW classifier — always returns 0.0 (safe).
    Replace with a real ML model later.
    """
    return 0.0

def load_image_from_url(url: str):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    return np.array(img)

def check_billboard_size(image: np.ndarray):
    h, w, _ = image.shape
    aspect = w / h
    if h > 2000 or w > 2000:  # arbitrary threshold
        return False, "Billboard too large"
    if aspect < 0.5 or aspect > 2.0:
        return False, "Billboard aspect ratio invalid"
    return True, None

import easyocr
reader = easyocr.Reader(["en"])

def check_text_compliance(image: np.ndarray):
    try:
        results = reader.readtext(image)
        detected_texts = [res[1] for res in results]
        print("Detected:", detected_texts)
        if not any("LIC" in t.upper() for t in detected_texts):
            return False, "Missing license text"
        return True, None
    except Exception as e:
        return False, f"OCR failed: {str(e)}"



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
    # 1) Quick URL reachability check
    try:
        r = requests.head(image_url, timeout=5)
        if r.status_code >= 400:
            return {"status": "error", "message": "Image URL not accessible"}
    except Exception:
        return {"status": "error", "message": "Image URL check failed"}

    # 2) Geofence & time window
    ok_geo, geo_msg = check_geofence(lat, lon)
    ok_time, time_msg = check_time_window(datetime.now(timezone.utc))

    violations = []

    try:
        img = load_image_from_url(image_url)

        ok_size, size_msg = check_billboard_size(img)
        ok_text, text_msg = check_text_compliance(img)

        if not ok_size:
            violations.append(size_msg)
        if not ok_text:
            violations.append(text_msg)

    except Exception as e:
        violations.append(f"ML check failed: {str(e)}")

    # 3) QR HMAC verification (optional)
    if qr_value:
        ok_qr, qr_msg = verify_qr_hmac(qr_value)
    else:
        ok_qr, qr_msg = True, None   # ✅ treat missing QR as allowed

    # 4) Content policy (NSFW) - stub for now
    nsfw_threshold = 0.80
    try:
        nsfw_score = nsfw_prob_stub(image_url)  # replace with real model later
    except Exception:
        nsfw_score = 0.0  # fail-open

    # 5) Collect violations
    if not ok_geo: violations.append(geo_msg)
    if not ok_time: violations.append(time_msg)
    if not ok_qr: violations.append(qr_msg)
    if nsfw_score >= nsfw_threshold: violations.append("NSFW content")

    if violations:
        status = "violation"
        message = "; ".join([m for m in violations if m])
    else:
        status = "success"
        message = "All checks passed"

    # 6) Server-side update of Supabase row (if report_id provided and service key set)
    if supabase and report_id:
        try:
            supabase.table("reports").update({"status": status, "message": message}).eq("id", report_id).execute()
        except Exception:
            # Don't fail the API response if the update fails; log in real deployment
            pass
    return {"status": status, "message": message}
