from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import os

app = FastAPI(title="Customer Purchase Propensity API")

model = None

class CustomerData(BaseModel):
    age: int
    income: float
    website_visits: int
    cart_adds: int
    time_on_site: float
    income_per_age: float
    active_user: int

@app.on_event("startup")
def load_model():
    global model
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("Purchase_Propensity")
        if experiment:
            runs = client.search_runs(experiment.experiment_id, order_by=["metrics.roc_auc DESC"])
            if runs:
                best_run = runs[0]
                model_uri = f"runs:/{best_run.info.run_id}/model"
                model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/predict")
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    
    return {"propensity": float(prediction[0])}

@app.get("/model-info")
def model_info():
    if model is None:
        return {"status": "Model not loaded"}
    return {"status": "Model loaded successfully"}
