# High-Level Design (HLD) Document
## DA5402 MLOps Final Project — Vehicle Insurance Cross-Sell Prediction

**Author:** Ganesh Mula | **Course:** DA5402, IIT Madras | **Version:** 1.0

---

## 1. System Overview

End-to-end MLOps system predicting vehicle insurance cross-sell interest. Covers data ingestion, training pipeline, experiment tracking, CI/CD deployment, and real-time inference.

---

## 2. High-Level Architecture

```mermaid
flowchart TD
    subgraph DataLayer["Data Layer"]
        A[(MongoDB Atlas\nProj1-Data Collection)]
    end

    subgraph TrainingLayer["Training Layer (orchestrated by Airflow)"]
        B[Data Ingestion\ntrain.csv / test.csv]
        C[Data Validation\nSchema Check]
        D[Data Transformation\nEncoding + Scaling + SMOTEENN]
        E[Model Trainer\nRandomForest / ExtraTrees / GradientBoosting]
        F[Model Evaluation\nNew vs Production F1]
        G[Model Pusher\nBest model to S3]
        B --> C --> D --> E --> F -->|Accepted| G
    end

    subgraph TrackingLayer["Tracking Layer"]
        H[MLflow\nSQLite backend\nParams + Metrics + Artifacts]
    end

    subgraph StorageLayer["Storage Layer"]
        I[(AWS S3\nmodel.pkl)]
    end

    subgraph DeploymentLayer["Deployment Layer"]
        J[GitHub Actions CI\nDocker Build + ECR Push]
        K[GitHub Actions CD\nEC2 Self-hosted Runner]
        L[AWS EC2\nDocker Container\nFastAPI Port 5000]
    end

    subgraph UILayer["User Interface Layer"]
        M[Browser\nHTML Form]
    end

    A --> B
    E --> H
    F --> H
    G --> I
    I --> L
    J --> K --> L
    L --> M
```

---

## 3. Five-Layer Architecture

```mermaid
flowchart LR
    L1[User Interface\nFastAPI + Jinja2]
    L2[Inference Layer\nVehicleDataClassifier]
    L3[Training Layer\n6-Stage Pipeline]
    L4[Storage Layer\nMongoDB + S3]
    L5[Deployment Layer\nDocker + ECR + EC2]

    L1 <-->|REST API| L2
    L2 <-->|Load model| L4
    L3 <-->|Read data / Push model| L4
    L5 -->|Hosts| L1
    L5 -->|Hosts| L2
```

---

## 4. Training Pipeline Flow

```mermaid
flowchart TD
    A([Start]) --> B[Data Ingestion\nMongoDB → CSV]
    B --> C{Schema Valid?}
    C -->|No| Z1([FAIL — Pipeline stops])
    C -->|Yes| D[Data Transformation\nGender encode, drop id\ndummies, scale, SMOTEENN]
    D --> E[Model Trainer\nTrain 3 models\nLog all to MLflow]
    E --> F{Train accuracy\n>= 0.60?}
    F -->|No| Z2([FAIL — Below threshold])
    F -->|Yes| G[Model Evaluation\nCompare new vs S3 model\nLog to MLflow]
    G --> H{F1 improvement\n>= 0.02?}
    H -->|No| Z3([REJECT — Keep production])
    H -->|Yes| I[Model Pusher\nUpload to S3]
    I --> J([SUCCESS])
```

---

## 5. CI/CD Pipeline

```mermaid
flowchart LR
    A[git push\nmain branch] --> B[GitHub Actions\nubuntu-latest]
    B --> C[Configure\nAWS credentials]
    C --> D[Login\nto ECR]
    D --> E[Build\nDocker image]
    E --> F[Push image\nto ECR latest]
    F --> G[Self-hosted runner\nEC2 Ubuntu]
    G --> H[Pull image\nfrom ECR]
    H --> I[Run container\nport 5000\nwith env vars]
```

---

## 6. Airflow DAG Structure

```mermaid
flowchart LR
    A[data_ingestion\nPythonOperator] --> B[data_validation\nPythonOperator]
    B --> C[data_transformation\nPythonOperator]
    C --> D[model_training\nPythonOperator]
    D --> E[model_evaluation\nPythonOperator]
    E --> F[model_pusher\nPythonOperator]

    note1[trigger_rule: all_success\nIf any task fails → downstream = upstream_failed]
```

---

## 7. Model Selection Logic

```mermaid
flowchart TD
    A[Load train/test arrays] --> B[Train RandomForest\nn=100, depth=10, entropy]
    A --> C[Train ExtraTrees\nn=100, depth=10, entropy]
    A --> D[Train GradientBoosting\nn=100, depth=10]
    B --> E[Log run to MLflow]
    C --> F[Log run to MLflow]
    D --> G[Log run to MLflow]
    E --> H{Compare F1 scores}
    F --> H
    G --> H
    H --> I[Select best model]
    I --> J[Log BestModel summary run\nto MLflow]
    J --> K[Wrap with preprocessor\nSave as model.pkl]
```

---

## 8. Technology Rationale

| Choice | Rationale |
|---|---|
| MongoDB Atlas | Cloud-native, flexible schema, easy bulk insert |
| RandomForest / ExtraTrees / GradientBoosting | All sklearn, no new dependencies, diverse ensemble approaches |
| SMOTEENN | Combined oversampling + cleaning — best for this imbalanced dataset |
| dill | Handles complex Python objects sklearn Pipeline can't serialize with pickle |
| MLflow SQLite | Reliable rendering in MLflow 3.x UI vs file-based mlruns |
| Airflow | Industry-standard pipeline orchestration with visual DAG |
| FastAPI | Async, high performance, auto OpenAPI docs |
| Docker + ECR + EC2 | Full reproducibility from dev to production |

---

## 9. Design Principles

**Loose Coupling:** Frontend ↔ Backend via REST only. Each pipeline stage independent.

**Single Responsibility:** One class per pipeline stage. Config/Artifact dataclasses separate concerns.

**Fail Fast:** `MyException` captures filename + line number. Airflow stops pipeline on any task failure.

**Reproducibility:** All hyperparameters in `constants/__init__.py`. MLflow logs every run. Docker ensures identical runtime.

---

## 10. Limitations & Future Work

| Limitation | Proposed Solution |
|---|---|
| No DVC data versioning | Integrate DVC with S3 remote |
| No Prometheus/Grafana | Add /metrics endpoint + Grafana dashboard |
| No docker-compose | Separate frontend/backend as two services |
| No unit tests | Add pytest per component |
| Manual trigger only | Schedule Airflow DAG with cron |
