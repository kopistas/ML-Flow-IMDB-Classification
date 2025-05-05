import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import time
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load the processed IMDB dataset"""
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/validation.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    return train_df, val_df, test_df

def evaluate_model(model, X, y, set_name):
    """Evaluate model and log metrics to MLflow"""
    y_pred = model.predict(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='binary')
    recall = recall_score(y, y_pred, average='binary')
    f1 = f1_score(y, y_pred, average='binary')
    
    # Log metrics to MLflow
    mlflow.log_metric(f"{set_name}_accuracy", accuracy)
    mlflow.log_metric(f"{set_name}_precision", precision)
    mlflow.log_metric(f"{set_name}_recall", recall)
    mlflow.log_metric(f"{set_name}_f1", f1)
    
    logger.info(f"{set_name} metrics:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def train_model():
    """Train IMDB sentiment classification model with MLflow tracking"""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("./mlruns")
    
    # Get experiment ID - use the experiment name we created in the entrypoint
    experiment = mlflow.get_experiment_by_name("imdb-sentiment")
    if experiment is None:
        logger.info("Creating experiment 'imdb-sentiment'")
        experiment_id = mlflow.create_experiment("imdb-sentiment")
    else:
        experiment_id = experiment.experiment_id
        
    logger.info(f"Using experiment ID: {experiment_id}")
    
    # Start an MLflow run
    with mlflow.start_run(run_name="imdb_sentiment_classifier", experiment_id=experiment_id) as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        # Load data
        logger.info("Loading data...")
        train_df, val_df, test_df = load_data()
        
        # Log dataset size as parameters
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("validation_size", len(val_df))
        mlflow.log_param("test_size", len(test_df))
        
        # Define pipeline parameters
        max_features = 10000
        C = 1.0
        
        # Log model parameters
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("classifier", "LogisticRegression")
        mlflow.log_param("C", C)
        
        # Create the pipeline
        logger.info("Creating and training model pipeline...")
        sentiment_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features)),
            ('classifier', LogisticRegression(C=C, max_iter=1000))
        ])
        
        # Train the model
        start_time = time.time()
        sentiment_pipeline.fit(train_df['text'], train_df['label'])
        training_time = time.time() - start_time
        
        # Log training time
        mlflow.log_metric("training_time_seconds", training_time)
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on train set
        logger.info("Evaluating on training set...")
        train_metrics = evaluate_model(sentiment_pipeline, train_df['text'], train_df['label'], "train")
        
        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        val_metrics = evaluate_model(sentiment_pipeline, val_df['text'], val_df['label'], "val")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = evaluate_model(sentiment_pipeline, test_df['text'], test_df['label'], "test")
        
        # Save the model
        logger.info("Saving model...")
        os.makedirs("models/artifacts", exist_ok=True)
        model_path = "models/artifacts/sentiment_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(sentiment_pipeline, f)
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(sentiment_pipeline, "sentiment_pipeline")
        
        # Log the model path
        mlflow.log_artifact(model_path)
        
        logger.info(f"Model saved to {model_path} and logged to MLflow")
        logger.info(f"MLflow run: {mlflow.get_artifact_uri()}")
        
        return sentiment_pipeline, run.info.run_id

if __name__ == "__main__":
    train_model() 