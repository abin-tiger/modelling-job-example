FROM python:3.9-slim

WORKDIR /app

RUN pip install pandas scikit-learn joblib --no-cache-dir

COPY train_model.py .

CMD ["python", "/app/train_model.py"]
