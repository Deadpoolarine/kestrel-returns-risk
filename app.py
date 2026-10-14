from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Load model and features
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

class OrderData(BaseModel):
    sales_channel: int
    payment_mode: int
    discount_pct: float
    qty: int
    order_value_inr: float
    promised_delivery_days: int
    delivery_pincode: int
    is_gift: int
    customer_prior_orders: int
    customer_prior_returns: int
    source: int
    city: int
    state: int
    shield_member: int
    family: int
    list_price_inr: float
    warranty_months: int

@app.post("/predict")
def predict(order: OrderData):
    data = pd.DataFrame([order.dict()])
    data = data[features] # Ensure order
    
    prob = model.predict_proba(data)[0][1]
    
    reasons = []
    if order.shield_member == 1:
        reasons.append("Customer is a Shield Member - DO NOT HOLD. High Lifetime Value.")
    if prob > 0.112:
        reasons.append(f"Return risk is {prob:.1%}. Recommend Rs 45 pre-dispatch call.")
    else:
        reasons.append(f"Low risk ({prob:.1%}). Safe to dispatch.")
        
    return {"risk_score": float(prob), "reasons": reasons}

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r") as f:
        return f.read()