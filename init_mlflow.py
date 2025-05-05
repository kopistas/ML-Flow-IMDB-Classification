import mlflow
import os

def init_mlflow():
    """Initialize MLflow experiment."""
    # Create mlruns directory if it doesn't exist
    os.makedirs("mlruns", exist_ok=True)
    
    # Set tracking URI
    mlflow.set_tracking_uri('./mlruns')
    
    # Create experiment if it doesn't exist
    experiment_name = "imdb-sentiment"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Creating MLflow experiment '{experiment_name}'")
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created experiment with ID: {experiment_id}")
    else:
        print(f"Experiment '{experiment_name}' already exists with ID: {experiment.experiment_id}")

if __name__ == "__main__":
    init_mlflow() 