import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.sklearn
import mlflow.xgboost

def train_models():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("Purchase_Propensity")
    
    # Read processed parquet
    files = glob.glob("data/processed/features.parquet/*.parquet")
    if not files:
        raise FileNotFoundError("Processed data not found.")
    
    df = pd.concat([pd.read_parquet(f) for f in files])
    
    X = df.drop(columns=["customer_id", "purchased"])
    y = df["purchased"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Logistic Regression
    with mlflow.start_run(run_name="Logistic_Regression"):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred_proba = lr.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(lr, "model")
        print(f"Logistic Regression AUC: {auc}")
        
    # Train XGBoost
    with mlflow.start_run(run_name="XGBoost"):
        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        xgb.fit(X_train, y_train)
        y_pred_proba = xgb.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_metric("roc_auc", auc)
        mlflow.xgboost.log_model(xgb, "model")
        print(f"XGBoost AUC: {auc}")

if __name__ == "__main__":
    train_models()
