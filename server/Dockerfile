FROM python:3.14-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code and the trained model only. The dataset is intentionally not included.
COPY main.py patient.py trained_model.pkl ./

EXPOSE 5000
CMD ["python", "main.py"]
