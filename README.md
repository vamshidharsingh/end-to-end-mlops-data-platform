# End-to-End MLOps & Data Engineering Platform

## 📖 Executive Overview
This **End-to-End MLOps & Data Engineering Platform** represents the pinnacle of machine learning operationalization. Built around the use case of **Customer Purchase Propensity Prediction**, this project demonstrates how to graduate from disorganized Jupyter Notebooks into a fully orchestrated, reproducible, and trackable enterprise-grade ML lifecycle. It integrates **Apache Airflow** for orchestration, **PySpark** for scalable ETL, **MLflow** for experiment tracking, and **FastAPI** for online inference.

## 🎯 Business Problem
Data Science teams frequently build accurate models locally, but struggle to deploy them reliably. Without orchestration, data drift goes unnoticed. Without experiment tracking, hyperparameter tuning is lost. Without automated ETL, data pipelines break. This platform solves the "Day 2 Operations" problem of Machine Learning by fully automating the pipeline from raw data ingestion to served API predictions.

## 🏗️ System Architecture & MLOps Lifecycle

```mermaid
graph TD
    subgraph Data Engineering Pipeline (PySpark)
        A[Raw CSV Data] --> B[Data Validation]
        B --> C[PySpark Transformation & Cleaning]
        C --> D[Feature Engineering]
        D --> E[(PostgreSQL Feature Store)]
    end
    
    subgraph MLOps Pipeline
        E --> F[Train/Test Split]
        F --> G[Train Logistic Regression Baseline]
        F --> H[Train XGBoost Classifier]
        G --> I[MLflow Tracking Server]
        H --> I
        I -->|Register Best Model| J[Model Registry]
    end
    
    subgraph Serving Layer
        J --> K[FastAPI Prediction Service]
        L[Client Request] --> K
        K --> M[Prediction Response]
    end
    
    N((Apache Airflow)) -.->|Orchestrates| B
    N -.->|Orchestrates| C
    N -.->|Orchestrates| G
    N -.->|Orchestrates| H
```

## 🧠 Platform Components

### 1. Data Engineering (PySpark)
Using PySpark allows the ETL pipeline to scale to massive datasets. The transformation layer cleans null values, handles categorical encodings, and engineers features such as `days_since_last_purchase`, `session_duration_aggregates`, and `engagement_scores`. The final curated features are pushed to PostgreSQL.

### 2. Orchestration (Apache Airflow)
Airflow manages the Directed Acyclic Graph (DAG) named `ml_pipeline`. It ensures that models are only trained *after* the ETL job successfully finishes. If data validation fails, the pipeline halts and alerts the team, preventing corrupted data from degrading the model.

### 3. Experiment Tracking (MLflow)
Every execution of the training script logs metrics to a local MLflow server. 
*   **Parameters Tracked:** Learning rate, max depth, estimators.
*   **Metrics Tracked:** ROC-AUC, Precision, Recall, F1-Score.
*   **Artifacts:** The serialized model artifact (`.pkl` / `.joblib`) is saved directly alongside the run for perfect reproducibility.

### 4. Serving (FastAPI)
A production-ready asynchronous API dynamically loads the best-performing model from MLflow's artifact store on startup. It exposes endpoints to predict purchase propensity for new, unseen customers in real-time.

## 🚀 Installation & Deployment

### Prerequisites
- Python 3.11+
- Docker & Docker Compose

### Local Environment Setup
1. **Clone & Environment:**
   ```bash
   git clone https://github.com/vamshidharsingh/end-to-end-mlops-data-platform.git
   cd end-to-end-mlops-data-platform
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Start the Infrastructure (Airflow, Postgres, MLflow):**
   ```bash
   docker-compose up -d
   ```
   *Note: Airflow might take 1-2 minutes to initialize its webserver.*

3. **Generate Synthetic Data:**
   ```bash
   python scripts/data_generator.py
   ```

4. **Access the UIs:**
   - **Airflow Web UI:** `http://localhost:8080` (Trigger the `ml_pipeline` DAG here)
   - **MLflow Tracking UI:** `http://localhost:5000` (View training metrics here)

5. **Start the API:**
   ```bash
   uvicorn api.main:app --port 8007 --reload
   ```

## 📡 API Documentation

### `POST /predict`
Predicts the likelihood of a customer purchasing.
**Payload:**
```json
{
  "recency": 12,
  "frequency": 4,
  "monetary_value": 150.50,
  "session_duration": 450,
  "device_type": "mobile"
}
```
**Response:**
```json
{
  "purchase_propensity": 0.82,
  "prediction": 1,
  "model_version": "xgboost_v2"
}
```

## 🧪 Testing & Evaluation
Tests are separated into Data Engineering tests (validating PySpark dataframe schemas and aggregation logic) and ML tests (verifying inference endpoint schemas).
```bash
pytest tests/
```

## 📁 Project Structure
```text
end-to-end-mlops-data-platform/
├── dags/
│   └── ml_pipeline.py     # Airflow DAG definition
├── src/
│   ├── transformation/    # PySpark ETL jobs
│   ├── training/          # XGBoost / LR MLflow training scripts
│   └── inference/         # Model loading utilities
├── api/                   # FastAPI endpoints
├── scripts/               # Data generation
├── tests/                 # Pytest suite
├── docker-compose.yml     # Infrastructure (Airflow, Postgres, MLflow)
├── requirements.txt
└── README.md
```

## 🔮 Future Enhancements
*   **EvidentlyAI Integration:** Adding automated data drift detection to monitor the distributions of incoming API requests.
*   **Kubeflow Kubernetes Deployment:** Upgrading from local Docker Compose to a distributed Kubernetes environment using Kubeflow.
*   **CI/CD Pipeline:** Adding GitHub Actions to automatically trigger tests and linters on Pull Requests.

## 📄 License
This project is licensed under the MIT License.
