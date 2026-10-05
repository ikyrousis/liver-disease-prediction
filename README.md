# AI Liver Disease Prediction

A client-server application in which a React web interface submits patient lab results
to a Flask backend that predicts liver disease with a probabilistic logic (**Problog**)
model trained on the Indian Liver Patient dataset. The server exposes a prediction
endpoint for the client: it binarizes each patient's values into boolean evidence terms,
evaluates them against rules learned from the training data, and returns the probability
of liver disease and of being healthy.

Built for the Hellenic Mediterranean University (Advanced Topics in Artificial
Intelligence), June 2023.

## Architecture

```
client (React, :3000)
        │  POST /liver-disease-prediction (JSON)
        ▼
server (Flask, :5000)
        │  binarizes lab values → boolean evidence atoms
        ▼
Problog model (trained from the Indian Liver Patient dataset)
```

| Component | Role |
|---|---|
| `server/` | Python/Flask backend: builds the Problog model on first start, serves predictions on `POST /liver-disease-prediction` |
| `client/` | React client: patient data form, displays the returned probabilities |

## Quick start

**Backend**:

```bash
cd server
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS and Linux: source .venv/bin/activate)
pip install -r requirements.txt
python main.py
```

On first start (or when `trained_model.pkl` is missing) the model is built from the
dataset and the server starts on port 5000. After that, startup simply loads the model.

**Frontend**:

```bash
cd client
npm install
npm start
```

Serves on port 3000. Open http://localhost:3000, fill in the patient form and submit.

**Docker:**:

```bash
docker compose up --build
```

Same URLs and API contract as the local run (client :3000, server :5000). The image runs
the trained model. To retrain inside the container, mount the dataset at
`/app/indian_liver_patient.csv` and comment out the `COPY` of `trained_model.pkl` in
`server/Dockerfile`, so the model is rebuilt at startup.

## Dataset

`indian_liver_patient.csv`, the [Indian Liver Patient Records](https://www.kaggle.com/datasets/uciml/indian-liver-patient-records)
(583 records, 10 features). Download it from Kaggle and place it **next to `main.py`**.
It is required only for retraining. The server
runs on the trained model without it.

## The model

- Patient lab values are binarized into boolean evidence terms (`male`, `young`,
  `normalBilirubin`, …). "Normal/young" means *at or below the training-split median*
  (data-derived cutoffs).
- One Problog annotated-disjunction rule per distinct evidence pattern:
  `w::liver_disease ; (1-w)::healthy :- <evidence>`, where `w` is the pattern's
  empirical class posterior in the training data.
- Inference uses Problog's evaluator with the patient's evidence atoms. Patterns never
  seen in training fall back to the training-set class prior. The API response flags
  this via `matched_training_pattern`.
- Training uses a stratified 70/30 split (pinned seed for reproducibility) and pickles
  the model together with its binarization thresholds.

## API

`POST /liver-disease-prediction`, request body (JSON):

```json
{"gender": "Male", "age": 62, "total_bilirubin": 10.9, "direct_bilirubin": 5.5,
 "alkaline_phosphotase": 699, "alamine_aminotransferase": 64,
 "aspartate_aminotransferase": 100, "total_proteins": 7.5, "albumin": 3.2,
 "albumin_and_globulin_ratio": 0.74}
```

Response (JSON):

```json
{"prediction": "Probability of liver disease: 1.0000\nProbability to be healthy: 0.0000\n",
 "probability_liver_disease": 1.0,
 "probability_healthy": 0.0,
 "matched_training_pattern": true}
```

## Model artifacts

Model training produces the following files:

| File | Description |
|---|---|
| `trained_model.pkl` | trained Problog model with binarization thresholds, loaded automatically at every server startup |
| `untrained_model.pl` | Problog program structure prior to weight estimation |
| `metrics.txt` | evaluation metrics of the held-out test set |

## Performance

| Metric | Score |
|---|---|
| Accuracy (30% test split) | 0.730 |
| Precision / Recall / F1 (macro) | 0.659 / 0.596 / 0.601 |
| Accuracy / balanced accuracy (5-fold CV) | 0.738 / 0.636 |

~0.73-0.76 is the practical ceiling for this dataset.
