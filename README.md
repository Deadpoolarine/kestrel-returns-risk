# Kestrel Home — Returns Risk Service

Pre-dispatch returns risk scoring API and operational decision engine for Kestrel Home Appliances.

## Overview & Business Logic
Instead of placing automated holds on risky orders (which causes a 12% cancellation rate), this service routes non-Shield orders with a >11.2% return probability to a Rs 45 pre-dispatch confirmation call queue. Kestrel Shield members are automatically exempted from holds to protect customer lifetime value.

Post-dispatch data leakage columns (`last_service_event_type` and `pickup_scheduled_at`) are stripped during training so the model relies strictly on pre-dispatch checkout signals.

## Quickstart (Clean Machine Setup)

No paid API keys are required. Follow these steps to run the service locally:

### 1. Install Dependencies
Open your terminal in the project folder and run:
pip install pandas scikit-learn fastapi uvicorn pydantic

### 2. Train Model & Generate Predictions
Place `train.csv`, `test_unlabelled.csv`, `customers.csv`, and `products.csv` in the root folder, then run:
python train.py
python predict.py

This generates the model artifacts (`model.pkl`, `encoders.pkl`, `features.pkl`) and outputs `predictions.csv`.

### 3. Start the Web Service
Run the FastAPI server:
uvicorn app:app --reload

### 4. Use the Application
* Web UI: Open http://127.0.0.1:8000 in your browser to test order payloads and view human-readable intervention reasons.
* API Endpoint: Send a POST request with a JSON order payload to http://127.0.0.1:8000/predict
