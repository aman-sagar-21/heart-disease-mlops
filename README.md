# Heart Disease Prediction using MLOps

## Project Overview

This project implements an end-to-end **Machine Learning Operations (MLOps)** pipeline for predicting the presence of heart disease using the UCI Heart Disease dataset.

The objective is not only to develop a machine learning model but also to demonstrate industry-standard MLOps practices, including:

- Data preprocessing and exploratory data analysis
- Feature engineering
- Model training and evaluation
- Experiment tracking using MLflow
- Model packaging
- REST API development using FastAPI
- Containerization using Docker
- Continuous Integration using GitHub Actions
- Deployment using Kubernetes
- Monitoring and logging

This project was developed as part of the MLOps coursework.

---

# Project Architecture

```
                    ┌───────────────────┐
                    │   Heart Disease   │
                    │      Dataset      │
                    └─────────┬─────────┘
                              │
                              ▼
                     Data Cleaning & EDA
                              │
                              ▼
                   Feature Engineering
                              │
                              ▼
                   Random Forest Model
                              │
                              ▼
                     MLflow Tracking
                              │
                              ▼
                     Model Packaging
                              │
                              ▼
                      FastAPI Service
                              │
                              ▼
                      Docker Container
                              │
                              ▼
                     Kubernetes Cluster
                              │
                              ▼
                 Prediction & Monitoring
```

---

# Dataset

**Dataset:** UCI Heart Disease Dataset

The dataset contains clinical information collected from patients for predicting the presence of heart disease.

### Features

| Feature | Description |
|----------|-------------|
| age | Age of patient |
| sex | Gender |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting ECG |
| thalach | Maximum heart rate |
| exang | Exercise induced angina |
| oldpeak | ST depression |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels |
| thal | Thalassemia |
| target | Heart disease diagnosis |

---

# Exploratory Data Analysis

The following analyses were performed:

- Dataset overview
- Missing value analysis
- Statistical summary
- Correlation analysis
- Distribution plots
- Target class distribution

---

# Machine Learning Pipeline

The preprocessing pipeline consists of:

- Missing value imputation
- Feature scaling using StandardScaler
- Model training using Scikit-Learn Pipeline

Two models were evaluated:

- Logistic Regression
- Random Forest Classifier

---

# Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------|----------|-----------|--------|----------|----------|
| Logistic Regression | 86.89% | 81.25% | 92.86% | 86.67% | 95.13% |
| **Random Forest** | **90.16%** | **86.67%** | **92.86%** | **89.66%** | **95.78%** |

The **Random Forest Classifier** achieved the best overall performance and was selected for deployment.

---

# Cross Validation

To ensure that the model generalizes well to unseen data, K-Fold Cross Validation was performed.

Cross-validation helps:

- Reduce overfitting
- Measure model stability
- Evaluate generalization performance
- Provide a more reliable estimate of model accuracy

---

# Experiment Tracking

MLflow was used for experiment tracking.

The following artifacts were logged:

- Hyperparameters
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Trained Model

---

# Project Structure

```
Assignment1/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
│
├── data/
│
├── deployment/
│   ├── deployment.yaml
│   └── service.yaml
│
├── models/
│   └── heart_disease_rf_mlflow_pipeline_v1.pkl
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Model_Development.ipynb
│   └── Experiment_Tracking_MLflow.ipynb
│
├── src/
│   └── predict.py
│
├── tests/
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .dockerignore
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/aman-sagar-21/heart-disease-mlops.git

cd heart-disease-mlops
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the FastAPI Application

Start the API

```bash
uvicorn app.main:app --reload
```

Application

```
http://127.0.0.1:8000
```

Swagger Documentation

```
http://127.0.0.1:8000/docs
```

Health Check

```
http://127.0.0.1:8000/health
```

---

# Example Prediction Request

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 2,
  "ca": 0,
  "thal": 3
}
```

Example Response

```json
{
    "prediction": "Heart Disease Detected",
    "confidence": 0.96
}
```

---

# Docker

Build Docker Image

```bash
docker build -t heart-disease-api .
```

Run Docker Container

```bash
docker run -p 8000:8000 heart-disease-api
```

---

# Kubernetes Deployment

Deploy the application

```bash
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
```

Verify deployment

```bash
kubectl get deployments

kubectl get pods

kubectl get services
```

Port forwarding

```bash
kubectl port-forward service/heart-disease-service 8000:80
```

---

# Continuous Integration

GitHub Actions was configured to automatically:

- Install project dependencies
- Execute unit tests using Pytest
- Validate the trained model
- Ensure successful builds on every push and pull request

Workflow file

```
.github/workflows/ci.yml
```

---

# Unit Testing

The project contains automated unit tests for:

- Model prediction
- Data preprocessing

Run tests

```bash
pytest
```

---

# Logging & Monitoring

The FastAPI application logs:

- Incoming prediction requests
- Prediction outputs
- Confidence scores
- Application startup events

Health endpoint

```
GET /health
```

returns the application status and can be used for monitoring.

---

# Technologies Used

- Python 3.12
- Pandas
- NumPy
- Scikit-Learn
- MLflow
- Joblib
- FastAPI
- Uvicorn
- Pytest
- Docker
- Kubernetes
- GitHub Actions

---

# Challenges Faced

During development, several practical MLOps challenges were encountered and resolved:

- Handling missing values in clinical data
- Selecting the best-performing classification model
- Managing MLflow dependency conflicts
- Resolving Scikit-Learn version compatibility issues with serialized pipelines
- Fixing Docker build failures caused by Windows-specific dependencies
- Configuring GitHub Actions for automated testing
- Deploying the application to Kubernetes using Docker Desktop
- Implementing structured logging for monitoring

These challenges provided valuable experience in building production-ready machine learning systems.

---

# Future Improvements

Potential enhancements include:

- Automated model retraining
- Model registry integration
- Prometheus & Grafana monitoring
- CI/CD deployment automation
- Cloud deployment (AWS EKS / Azure AKS / Google GKE)
- Authentication and API security

---

# Author

**Aman Sagar**
**(2024ac05987)**
M.Tech – Artificial Intelligence & Machine Learning

MLOps Assignment

---

# License

This project was developed for academic purposes.