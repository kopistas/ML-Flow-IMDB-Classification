FROM python:3.9-slim

WORKDIR /app

# Install system dependencies needed for building Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add additional package for running the MLflow UI
RUN pip install --no-cache-dir datasets

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p data/processed models/artifacts mlruns

# Copy and set permissions for entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose ports: 5000 for the API and 5001 for MLflow UI
EXPOSE 5000 5001

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"] 