import unittest
import requests
import json
import os
import sys
import subprocess
import time
import signal
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Check if API is already running
        try:
            response = requests.get("http://localhost:5000/health")
            cls.api_running = response.status_code == 200
        except:
            cls.api_running = False
            
        if not cls.api_running:
            # Start the API server
            print("Starting API server for tests...")
            cls.api_process = subprocess.Popen(
                ["python", "app/api.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            # Wait for API to start
            time.sleep(5)
    
    @classmethod
    def tearDownClass(cls):
        if not cls.api_running:
            # Stop the API server
            print("Stopping API server...")
            cls.api_process.send_signal(signal.SIGINT)
            cls.api_process.wait()
    
    def test_health_endpoint(self):
        """Test the health endpoint"""
        response = requests.get("http://localhost:5000/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")
    
    def test_predict_single(self):
        """Test prediction with a single text input"""
        data = {"text": "This movie was fantastic and I loved every minute of it!"}
        response = requests.post("http://localhost:5000/predict", json=data)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        
        # Check that prediction contains expected fields
        self.assertIn("sentiment", result)
        self.assertIn("label", result)
        self.assertIn("text", result)
        
        # Check types
        self.assertIsInstance(result["sentiment"], str)
        self.assertIsInstance(result["label"], int)
        self.assertEqual(result["text"], data["text"])
    
    def test_predict_batch(self):
        """Test prediction with a batch of text inputs"""
        data = {"text": [
            "This movie was fantastic and I loved every minute of it!",
            "What a terrible waste of time and money."
        ]}
        
        response = requests.post("http://localhost:5000/predict", json=data)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        
        # Check that the result contains predictions field
        self.assertIn("predictions", result)
        
        # Check that we have the right number of predictions
        self.assertEqual(len(result["predictions"]), len(data["text"]))
        
        # Check each prediction
        for prediction in result["predictions"]:
            self.assertIn("sentiment", prediction)
            self.assertIn("label", prediction)
            self.assertIn("text", prediction)
            self.assertIsInstance(prediction["sentiment"], str)
            self.assertIsInstance(prediction["label"], int)

if __name__ == "__main__":
    unittest.main() 