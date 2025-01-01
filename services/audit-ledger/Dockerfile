FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ /app/shared/
COPY services/audit-ledger/ /app/
ENV PYTHONPATH=/app
EXPOSE 8011
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011"]
