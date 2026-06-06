# Return Prediction & Fraud Detection Hub

Predict whether placed orders will be returned or flagged as fraudulent using Machine Learning techniques. This application compares traditional supervised classifiers (with and without SMOTE resampling) side-by-side with unsupervised anomaly detection models (Isolation Forest, One-Class SVM).

## Features

- **Memory-Efficient Ingestion**: Reads the large transaction dataset in chunks to avoid running out of RAM.
- **Data Synthesis**: Models both fraud target (`isFraud` from original dataset) and return target (`isReturned` generated via product category/amount metrics).
- **Class Imbalance Handling**: Contrasts performance differences between default training and SMOTE-based class balancing.
- **Anomaly Detection**: Compares supervised learning with unsupervised Isolation Forest and One-Class SVM anomaly detectors.
- **Interactive UI Dashboard**: Streamlit dashboard showing:
  - Metrics comparison table.
  - Interactive ROC curves.
  - Interactive Confusion Matrices.
  - Live simulation form for custom transaction prediction.

---

## Setup & Running Locally

### 1. Initialize Virtual Environment

Ensure you are in the workspace folder (`c:\Users\s4shi\Desktop\A fraud detection`) and run:

```bash
# Create the virtual environment
python -m venv venvdet

# Activate the virtual environment
# Windows (PowerShell):
.\venvdet\Scripts\Activate.ps1
# Windows (CMD):
.\venvdet\Scripts\activate.bat
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Training Pipeline

This script draws a representative sample (~20,000 records), merges the transaction and identity files, trains the models, and serializes the preprocessing pipeline and trained models into the `models/` folder.

```bash
python src/train.py
```

### 4. Launch Streamlit Application

```bash
streamlit run app.py
```
Open your browser at the address printed in terminal (default `http://localhost:8501`).

---

## Deploying with Docker

### Using Docker Compose (Recommended)

1. Make sure **Docker Desktop** is running.
2. Spin up the container:

```bash
docker compose up --build
```
This builds the container image and maps:
- `models/` directory (to share trained models between the host and container).
- `data/` directory (to make datasets available for training).
- Streamlit application port to `http://localhost:8501`.
