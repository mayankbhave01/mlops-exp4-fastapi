# MLOps Experiment 4

## FastAPI ML API

This project demonstrates deployment of a Machine Learning model using FastAPI.

### Features
- FastAPI REST API
- Iris dataset prediction
- Swagger UI testing

### Run locally

Install dependencies:

pip install -r requirements.txt

Run server:

python -m uvicorn app.main:app --reload

Open browser:

http://127.0.0.1:8000/docs

### Example Prediction Input

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
