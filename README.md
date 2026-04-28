# DA5402 — MLOps Final Project
### Vehicle Insurance Cross-Sell Prediction

<!-- [![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](https://mlflow.org/)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20EC2%20%7C%20ECR-yellow)](https://aws.amazon.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black)](https://github.com/features/actions) -->

---

## 📌 Problem Statement

Predict whether an existing health insurance customer would be interested in purchasing **vehicle insurance** offered by the same company. This is a binary classification problem (Response: 1 = Interested, 0 = Not Interested).

---

## 🗂️ Project Structure

```
DA5402-ML_Ops_Final_Project/
│
├── .github/
│   └── workflows/
│       └── aws.yaml                  # CI/CD pipeline (GitHub Actions)
│
├── config/
│   ├── schema.yaml                   # Dataset schema for validation
│   └── model.yaml                    # Model configuration
│
├── notebook/
│   ├── mongoDB_demo.ipynb            # MongoDB data push notebook
│   ├── exp-notebook.ipynb            # EDA & Feature Engineering
│   └── data.csv                      # Raw dataset
│
├── src/
│   ├── cloud_storage/
│   │   └── aws_storage.py            # S3 upload/download helpers
│   ├── components/
│   │   ├── data_ingestion.py         # MongoDB → CSV
│   │   ├── data_validation.py        # Schema validation
│   │   ├── data_transformation.py    # Feature engineering + SMOTEENN
│   │   ├── model_trainer.py          # RandomForest training + MLflow logging
│   │   ├── model_evaluation.py       # New vs production model comparison
│   │   └── model_pusher.py           # Push best model to S3
│   ├── configuration/
│   │   ├── mongo_db_connection.py    # MongoDB Atlas connection
│   │   └── aws_connection.py         # AWS S3 connection
│   ├── constants/
│   │   └── __init__.py               # All project-wide constants
│   ├── data_access/
│   │   └── proj1_data.py             # MongoDB → DataFrame
│   ├── entity/
│   │   ├── config_entity.py          # Dataclasses for component configs
│   │   ├── artifact_entity.py        # Dataclasses for component outputs
│   │   ├── estimator.py              # MyModel: preprocessor + classifier
│   │   └── s3_estimator.py           # S3 model pull/push
│   ├── exception/
│   │   └── __init__.py               # Custom exception with traceback
│   ├── logger/
│   │   └── __init__.py               # Rotating file + console logger
│   ├── pipline/
│   │   ├── training_pipeline.py      # End-to-end training pipeline
│   │   └── prediction_pipeline.py    # Inference pipeline
│   └── utils/
│       └── main_utils.py             # YAML, dill, numpy utilities
│
├── static/css/                       # Frontend CSS
├── templates/
│   └── vehicledata.html              # Jinja2 prediction form
│
├── app.py                            # FastAPI application
├── demo.py                           # Pipeline test script
├── Dockerfile                        # Docker container config
├── requirements.txt                  # Python dependencies
├── setup.py                          # Local package installer
└── pyproject.toml                    # Build system config
```

---

## 🏗️ Architecture & Pipeline Flow

```
MongoDB Atlas
     │
     ▼
Data Ingestion ──► Data Validation ──► Data Transformation
                                              │
                                              ▼
                                       Model Trainer ◄── MLflow Tracking
                                              │
                                              ▼
                                      Model Evaluation
                                    (New vs S3 Production)
                                              │
                                    ┌─────────┴─────────┐
                                  Accept              Reject
                                    │
                                    ▼
                              Model Pusher
                            (Upload to S3)
                                    │
                                    ▼
                         FastAPI Prediction App
                         (EC2 via Docker/ECR)
```

---

## 📊 Dataset

| Feature | Type | Description |
|---|---|---|
| Gender | Categorical | Male / Female |
| Age | Numerical | Age of customer |
| Driving_License | Binary | 0 = No, 1 = Yes |
| Region_Code | Numerical | Region of customer |
| Previously_Insured | Binary | 0 = No, 1 = Yes |
| Vehicle_Age | Categorical | < 1 Year / 1-2 Year / > 2 Years |
| Vehicle_Damage | Categorical | Yes / No |
| Annual_Premium | Numerical | Annual premium amount |
| Policy_Sales_Channel | Numerical | Channel code |
| Vintage | Numerical | Days associated with company |
| **Response** | **Target** | **0 = Not Interested, 1 = Interested** |

- **Total Records:** 381,109
- **Class Imbalance handled via:** SMOTEENN

---

## 🤖 Model

- **Algorithm:** Random Forest Classifier
- **Hyperparameters:**
  - `n_estimators`: 200
  - `max_depth`: 10
  - `min_samples_split`: 7
  - `min_samples_leaf`: 6
  - `criterion`: entropy
  - `random_state`: 101

### Performance Metrics (Latest Run)

| Metric | Score |
|---|---|
| Accuracy | 0.9243 |
| F1 Score | 0.9320 |
| Precision | 0.8812 |
| Recall | 0.9889 |

---

## 📈 MLflow Experiment Tracking

MLflow is integrated into the model training component to track:
- **Hyperparameters:** all 6 RandomForest parameters
- **Metrics:** accuracy, F1, precision, recall
- **Model artifact:** logged via `mlflow.sklearn.log_model`
- **Experiment name:** `VehicleInsurance-RandomForest`

To launch the MLflow UI:
```bash
mlflow ui --port 5001 --backend-store-uri ./mlruns
```
Then open: `http://127.0.0.1:5001/#/experiments/1`

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10
- MongoDB Atlas account
- AWS account (S3, EC2, ECR, IAM)
- Docker Desktop

### 1. Clone the Repository
```bash
git clone https://github.com/Ganesh2024/DA5402-ML_Ops_Final_Project.git
cd DA5402-ML_Ops_Final_Project
```

### 2. Create Virtual Environment
```bash
conda create -n vehicle python=3.10 -y
conda activate vehicle
pip install -r requirements.txt
```

### 3. Set Environment Variables

**PowerShell:**
```powershell
$env:MONGODB_URL = "mongodb+srv://<username>:<password>@<cluster>.mongodb.net/"
$env:AWS_ACCESS_KEY_ID = "<your-access-key>"
$env:AWS_SECRET_ACCESS_KEY = "<your-secret-key>"
```

**Bash:**
```bash
export MONGODB_URL="mongodb+srv://<username>:<password>@<cluster>.mongodb.net/"
export AWS_ACCESS_KEY_ID="<your-access-key>"
export AWS_SECRET_ACCESS_KEY="<your-secret-key>"
```

### 4. Run the Application
```bash
python app.py
```
App runs at: `http://localhost:5000`

---

## 🚀 CI/CD Pipeline (GitHub Actions + AWS)

### Infrastructure
- **ECR:** Stores Docker image (`vehicleproj`)
- **EC2:** Ubuntu 24.04, T2 Medium — runs the Docker container
- **Self-hosted Runner:** EC2 acts as GitHub Actions runner for CD

### Pipeline Stages
```
git push → GitHub Actions triggered
    │
    ├── Continuous-Integration (ubuntu-latest)
    │       ├── Checkout code
    │       ├── Configure AWS credentials
    │       ├── Login to ECR
    │       └── Build & Push Docker image to ECR
    │
    └── Continuous-Deployment (self-hosted EC2)
            ├── Checkout code
            ├── Configure AWS credentials
            ├── Login to ECR
            └── Pull & Run Docker container on port 5000
```

### Required GitHub Secrets
```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
ECR_REPO
MONGODB_URL
```

---

## 🐳 Docker

### Build Locally
```bash
docker build -t vehicle-insurance .
```

### Run Locally
```bash
docker run -d -p 5000:5000 \
  -e MONGODB_URL=<url> \
  -e AWS_ACCESS_KEY_ID=<key> \
  -e AWS_SECRET_ACCESS_KEY=<secret> \
  vehicle-insurance
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Renders prediction form |
| POST | `/` | Submits form and returns prediction |
| GET | `/train` | Triggers full training pipeline |

---

## ☁️ AWS Services Used

| Service | Purpose |
|---|---|
| S3 | Store trained model artifact (`model.pkl`) |
| EC2 | Host the FastAPI application via Docker |
| ECR | Store Docker image for deployment |
| IAM | Manage access credentials |

---

## 📦 Key Dependencies

```
fastapi, uvicorn          # Web framework
scikit-learn, imblearn    # ML + SMOTEENN
pymongo, certifi          # MongoDB connection
boto3, botocore           # AWS S3 interaction
mlflow                    # Experiment tracking
dill                      # Model serialization
PyYAML                    # Config file parsing
jinja2                    # HTML templating
from_root                 # Path resolution
```

---

## 👤 Author

**Ganesh Mula**

- Roll No: da25m019 
- Course: DA5402 — ML Operations

---


