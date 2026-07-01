FROM python:3.12.7-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY MLProject/conda.yaml .
RUN pip install --no-cache-dir \
    scikit-learn==1.5.2 \
    pandas==2.2.3 \
    numpy==2.1.3 \
    mlflow==2.19.0 \
    matplotlib==3.9.2 \
    seaborn==0.13.2

# Copy application code
COPY MLProject/ .

# Expose MLflow serving port
EXPOSE 5000

# Default command
CMD ["python", "modelling.py"]
