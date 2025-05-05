import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def start_mlflow_server():
    """Start the MLflow tracking server"""
    # Create directory for MLflow artifacts if it doesn't exist
    os.makedirs("mlruns", exist_ok=True)
    
    # Start the MLflow server
    logger.info("Starting MLflow server...")
    
    cmd = [
        "mlflow", "ui",
        "--backend-store-uri", "./mlruns",
        "--host", "0.0.0.0",
        "--port", "5001"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run the MLflow server
    process = subprocess.Popen(cmd)
    logger.info(f"MLflow server started with PID {process.pid}")
    logger.info("MLflow UI available at http://localhost:5001")
    
    # Keep the server running until interrupted
    try:
        process.wait()
    except KeyboardInterrupt:
        logger.info("Stopping MLflow server...")
        process.terminate()
        process.wait()
        logger.info("MLflow server stopped")

if __name__ == "__main__":
    start_mlflow_server() 