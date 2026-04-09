FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY inference_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY inference_service /app/inference_service
COPY lab3 /app/lab3

EXPOSE 8000

CMD ["uvicorn", "inference_service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
