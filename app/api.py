import os
import json
import pickle
import logging
import mlflow
from flask import Flask, request, jsonify
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model
def load_model():
    """Load the trained model"""
    # Try to load from local file first
    model_path = "models/artifacts/sentiment_model.pkl"
    
    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    # If local file doesn't exist, try to load from MLflow
    logger.info("Local model not found, trying to load from MLflow")
    mlflow.set_tracking_uri("./mlruns")
    
    try:
        # Get the experiment ID 
        experiment = mlflow.get_experiment_by_name("imdb-sentiment")
        if experiment is None:
            raise ValueError("Experiment 'imdb-sentiment' not found")
            
        experiment_id = experiment.experiment_id
        
        # Get the latest run from this experiment
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[experiment_id], 
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if len(runs) == 0:
            raise ValueError(f"No runs found for experiment {experiment_id}")
            
        run_id = runs[0].info.run_id
        
        logger.info(f"Loading model from MLflow run {run_id}")
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/sentiment_pipeline")
        
        return model
    except Exception as e:
        logger.warning(f"Failed to load model from MLflow: {e}")
        logger.info("Creating a simple fallback model")
        
        # Create a simple fallback model if nothing else is available
        # This ensures the API can start even without previous training
        fallback_model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', LogisticRegression(max_iter=100))
        ])
        # We'll train this on a tiny amount of data to make it functional
        # In a real-world app, you might want a more sophisticated solution
        fallback_model.fit(
            ["positive review", "negative review", "great movie", "terrible film"],
            [1, 0, 1, 0]
        )
        
        return fallback_model

# Load the model at startup
model = load_model()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions on input data"""
    if not request.json or 'text' not in request.json:
        return jsonify({"error": "Request must include 'text' field"}), 400
    
    # Get the text from the request
    text = request.json['text']
    
    if isinstance(text, list):
        # Handle batch predictions
        predictions = model.predict(text)
        sentiment = ['positive' if p == 1 else 'negative' for p in predictions]
        result = {'predictions': [{'text': t, 'sentiment': s, 'label': int(l)} 
                                  for t, s, l in zip(text, sentiment, predictions)]}
    else:
        # Handle single prediction
        prediction = model.predict([text])[0]
        sentiment = 'positive' if prediction == 1 else 'negative'
        result = {'text': text, 'sentiment': sentiment, 'label': int(prediction)}
    
    return jsonify(result)

if __name__ == '__main__':
    logger.info("Starting prediction API...")
    app.run(host='0.0.0.0', port=5000, debug=False) 