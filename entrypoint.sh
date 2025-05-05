#!/bin/bash
set -e

# Create MLflow experiment first
echo "Initializing MLflow experiment..."
python init_mlflow.py

# Start the MLflow server in the background
echo "Starting MLflow server..."
python mlflow_server/run_server.py &
sleep 2

# Download and prepare the data if needed
if [ ! -f "data/processed/train.csv" ]; then
    echo "Preparing data..."
    python data/prepare_data.py
fi

# Train the model if it doesn't exist
if [ ! -f "models/artifacts/sentiment_model.pkl" ]; then
    echo "Training model..."
    python models/train.py
fi

# Start the API server
echo "Starting API server..."
python app/api.py 