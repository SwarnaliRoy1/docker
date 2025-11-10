### Iris Classifier API 🌸

Container-ready FastAPI service that serves a trained scikit-learn Iris model. Comes with a Dockerfile and an example GitHub Actions workflow to build and push the image automatically.

This repo is a minimal, end-to-end demo for:

shipping a trained model (model.joblib)

serving it through a lightweight FastAPI app (iris_fastapi.py)

building and pushing Docker images to Google Artifact Registry via GitHub Actions

deploying to Google Kubernetes Engine (GKE) right from GitHub Actions

Perfect for MLOps walk-throughs, CI/CD experiments, or simply proving “yes, I can deploy an ML model.” 😎

## What it does

Loads a pre-trained Iris classifier (model.joblib)

Starts a FastAPI server

Exposes a prediction endpoint (likely /predict) that takes sepal/petal features and returns the predicted class

Can be run with uvicorn directly or via Docker

## Run locally

Create & activate venv (optional but nice):

python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate


Install deps:

pip install -r req.txt


Run the API (assuming the app is called app inside iris_fastapi.py):

uvicorn iris_fastapi:app --host 0.0.0.0 --port 8000 --reload


Open in browser:

Swagger UI: http://127.0.0.1:8000/docs

OpenAPI JSON: http://127.0.0.1:8000/openapi.json

## Example request

you can hit it like this:

curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'


Expected response:

{
  "prediction": "Iris-setosa",
  "class_id": 0
}

## If you want to reproduce the setup, just open workflow.pdf and follow the steps in order.
