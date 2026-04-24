"""
FastAPI Loan Defaulter Prediction App
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import joblib
import os

# ── Load model artifacts ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError(
        "model.pkl / scaler.pkl not found.\n"
        "Please run  python train_model.py  first."
    )

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ── FastAPI setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="Loan Defaulter Predictor")

# Ensure the 'static' folder exists or this line will crash
static_path = os.path.join(BASE_DIR, "static")
if not os.path.exists(static_path):
    os.makedirs(static_path)

app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Education mapping (must match training)
EDU_MAP = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}

# Housing one-hot (must match training column order)
HOUSING_MAP = {
    "Mortgage": (0, 0),   # baseline
    "Own":      (1, 0),
    "Rent":     (0, 1),
}

def _risk_level(prob: float) -> str:
    if prob < 30:   return "Low"
    if prob < 60:   return "Medium"
    if prob < 80:   return "High"
    return "Very High"

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: float               = Form(...),
    income: float            = Form(...),
    loan_amount: float       = Form(...),
    credit_score: float      = Form(...),
    employment_years: float  = Form(...),
    education_level: str     = Form(...),
    housing_status: str      = Form(...),
):
    # 1. Encode categorical
    edu_encoded = EDU_MAP.get(education_level, 1)
    housing_own, housing_rent = HOUSING_MAP.get(housing_status, (0, 0))

    # 2. Build feature names list (must match train_model.py order)
    feature_names = [
    "Age", "Income", "Loan_Amount", "Credit_Score",
    "Employment_Years", "Education_Level",
    "Housing_Status_Own", "Housing_Status_Rent"
    ]
features_df = pd.DataFrame([[
    data.age, data.income, data.loan_amount, data.credit_score,
    data.employment_years, edu_encoded, housing_own, housing_rent
]], columns=feature_names)

features_scaled = scaler.transform(features_df)

    # 3. Create DataFrame to avoid UserWarning about feature names
    features_df = pd.DataFrame([[
        age, income, loan_amount, credit_score,
        employment_years, edu_encoded, housing_own, housing_rent
    ]], columns=feature_names)

    # 4. Scale & Predict
    features_scaled = scaler.transform(features_df)
    prediction   = model.predict(features_scaled)[0]
    proba        = model.predict_proba(features_scaled)[0]
    
    default_prob = round(float(proba[1]) * 100, 2)
    safe_prob    = round(float(proba[0]) * 100, 2)

    result = {
        "is_defaulter"  : bool(prediction),
        "default_prob"  : default_prob,
        "safe_prob"     : safe_prob,
        "label"         : "⚠️ LIKELY DEFAULTER" if prediction else "✅ NOT A DEFAULTER",
        "risk_level"    : _risk_level(default_prob),
    }

    # Echo back form values for UI persistence
    form_data = {
        "age": age, "income": income, "loan_amount": loan_amount,
        "credit_score": credit_score, "employment_years": employment_years,
        "education_level": education_level, "housing_status": housing_status,
    }

    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "result": result, "form_data": form_data}
    )

# ── REST API endpoint (JSON) ───────────────────────────────────────────────────
from pydantic import BaseModel

class CustomerData(BaseModel):
    age: float
    income: float
    loan_amount: float
    credit_score: float
    employment_years: float
    education_level: str
    housing_status: str

@app.post("/api/predict")
async def api_predict(data: CustomerData):
    edu_encoded = EDU_MAP.get(data.education_level, 1)
    housing_own, housing_rent = HOUSING_MAP.get(data.housing_status, (0, 0))

    features = np.array([[
        data.age, data.income, data.loan_amount, data.credit_score,
        data.employment_years, edu_encoded, housing_own, housing_rent,
    ]])
    
    features_scaled = scaler.transform(features)
    prediction      = int(model.predict(features_scaled)[0])
    proba           = model.predict_proba(features_scaled)[0]

    return {
        "prediction"   : prediction,
        "is_defaulter" : bool(prediction),
        "default_probability": round(float(proba[1]) * 100, 2),
        "safe_probability"   : round(float(proba[0]) * 100, 2),
        "risk_level"   : _risk_level(round(float(proba[1]) * 100, 2)),
    }
