# 🤖 Ad Violation Backend

![FastAPI](https://img.shields.io/badge/FastAPI-109989?logo=fastapi&logoColor=white&style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?logo=supabase&logoColor=white&style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)

The **Ad Violation Backend** is a **FastAPI server** that acts as the central processing unit for your ad violation reporting system.  
It receives photo and location data from the mobile app, runs a series of checks using powerful AI, and logs the results to your **Supabase database**.

---

## ✨ Features

- **Image Analysis with Gemini**  
  Uses the **Gemini API** to analyze images for:  
  - Billboard detection  
  - Text extraction  
  - Sensitive content filtering  

- **Geolocation & Time Fencing**  
  Validates if a report was submitted from an **authorized geographic zone** and within a **specific time window**.  

- **QR Code Verification**  
  Authenticates QR codes using **HMAC signatures** to confirm their validity.  

- **Supabase Integration**  
  Updates the `reports` table with the results of AI analysis, providing a **centralized and reliable record**.  

- **Clean REST API**  
  Exposes a simple **endpoint** for the mobile app to interact with.  

---

## ⚙️ Prerequisites

Before running the backend, ensure you have:

- [Python 3.8+](https://www.python.org/downloads/)  
- pip (Python package installer)  

---

## 🚀 Installation

Clone the repository:

```bash
git clone [your-repo-url]
cd ad_violation_backend
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root with the following keys:

```env
SUPABASE_URL="https://[your-project-id].supabase.co"
SUPABASE_SERVICE_ROLE_KEY="[your-supabase-service-role-key]"
QR_HMAC_SECRET="[your-secret-key-for-qrs]"
GOOGLE_API_KEY="[your-gemini-api-key]"
```

---

## ▶️ Running the Server

Start the development server with:

```bash
uvicorn main:app --reload
```

The server will run at:  
👉 `http://127.0.0.1:8000`  

API docs available at:  
👉 `http://127.0.0.1:8000/docs`  

---

## 📌 API Endpoint

### `POST /analyze`

**Description:**  
Analyzes a reported image and updates the Supabase database with results.  

**Parameters:**  
- `image_url` → Publicly accessible image URL  
- `lat` → Latitude of the report  
- `lon` → Longitude of the report  
- `qr_value` → (Optional) Scanned QR code value  
- `report_id` → Supabase row ID to update  

**Response (JSON):**

```json
{
  "status": "success | violation | error",
  "message": "Detailed outcome message"
}
```

---

## 📜 License

This project is licensed under the **MIT License**.  
