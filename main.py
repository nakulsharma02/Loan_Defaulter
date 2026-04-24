from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd
import joblib
import os

# ── Load model ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError("model.pkl or scaler.pkl not found.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ── App setup ─────────────────────────────────────────────
app = FastAPI()

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

static_path = os.path.join(BASE_DIR, "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")

# ── Mappings ─────────────────────────────────────────────
EDU_MAP = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}

HOUSING_MAP = {
    "Mortgage": (0, 0),
    "Own": (1, 0),
    "Rent": (0, 1),
}

def risk_level(prob):
    if prob < 30:
        return "Low"
    elif prob < 60:
        return "Medium"
    elif prob < 80:
        return "High"
    return "Very_High"

# ── Home ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ── HTML FORM ROUTE ─────────────────────────────────────────────
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: float = Form(...),
    income: float = Form(...),
    loan_amount: float = Form(...),
    credit_score: float = Form(...),
    employment_years: float = Form(...),
    education_level: str = Form(...),
    housing_status: str = Form(...)
):
    edu = EDU_MAP.get(education_level, 1)
    own, rent = HOUSING_MAP.get(housing_status, (0, 0))

    cols = [
        "Age", "Income", "Loan_Amount", "Credit_Score",
        "Employment_Years", "Education_Level",
        "Housing_Status_Own", "Housing_Status_Rent"
    ]

    df = pd.DataFrame([[
        age, income, loan_amount, credit_score,
        employment_years, edu, own, rent
    ]], columns=cols)

    scaled = scaler.transform(df)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0]

    default_prob = round(float(prob[1]) * 100, 2)
    safe_prob = round(float(prob[0]) * 100, 2)

    result = {
        "is_defaulter": bool(pred),
        "default_prob": default_prob,
        "safe_prob": safe_prob,
        "label": "⚠️ LIKELY DEFAULTER" if pred else "✅ NOT A DEFAULTER",
        "risk_level": risk_level(default_prob),
    }

    form_data = {
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "education_level": education_level,
        "housing_status": housing_status,
    }

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "form_data": form_data}
    )

# ── API ROUTE (FORM BASED) ─────────────────────────────────────────────
@app.post("/api/predict")
async def api_predict(
    age: float = Form(...),
    income: float = Form(...),
    loan_amount: float = Form(...),
    credit_score: float = Form(...),
    employment_years: float = Form(...),
    education_level: str = Form(...),
    housing_status: str = Form(...)
):
    edu = EDU_MAP.get(education_level, 1)
    own, rent = HOUSING_MAP.get(housing_status, (0, 0))

    cols = [
        "Age", "Income", "Loan_Amount", "Credit_Score",
        "Employment_Years", "Education_Level",
        "Housing_Status_Own", "Housing_Status_Rent"
    ]

    df = pd.DataFrame([[
        age, income, loan_amount, credit_score,
        employment_years, edu, own, rent
    ]], columns=cols)

    scaled = scaler.transform(df)

    pred = int(model.predict(scaled)[0])
    prob = model.predict_proba(scaled)[0]

    return {
        "prediction": pred,
        "is_defaulter": bool(pred),
        "default_probability": round(float(prob[1]) * 100, 2),
        "safe_probability": round(float(prob[0]) * 100, 2),
        "risk_level": risk_level(round(float(prob[1]) * 100, 2)),
    }
