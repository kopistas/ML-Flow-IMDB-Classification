import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset

def download_and_prepare_imdb():
    """Download IMDB dataset and prepare for training"""
    print("Downloading IMDB dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Load the IMDB dataset
    imdb = load_dataset("imdb")
    
    # Convert to pandas for easier handling
    train_df = pd.DataFrame(imdb["train"])
    test_df = pd.DataFrame(imdb["test"])
    
    # Split training data into train and validation
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # Save to CSV
    print("Saving processed datasets...")
    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/validation.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    
    print(f"Train set: {len(train_df)} examples")
    print(f"Validation set: {len(val_df)} examples")
    print(f"Test set: {len(test_df)} examples")
    print("Data preparation completed!")

if __name__ == "__main__":
    download_and_prepare_imdb() 