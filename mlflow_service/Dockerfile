FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow==2.22.0 \
    psycopg2-binary==2.9.10 \
    boto3==1.42.1

RUN mkdir -p /mlflow_artifacts

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "postgresql+psycopg2://multimodal:multimodal@postgres:5432/multimodal_diffusion", "--artifacts-destination", "/mlflow_artifacts", "--serve-artifacts"]
