import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from src.utils import setup_logging, ensure_directory_exists

# Set up logging
logger = setup_logging('dataset_download')

def generate_synthetic_data(X, y, n_synthetic=1000):
    """Generate synthetic data points based on the original data."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    synthetic_data = []
    synthetic_labels = []
    
    # Generate synthetic samples for each class
    for class_label in np.unique(y):
        # Get samples for this class
        class_samples = X_scaled[y == class_label]
        
        # Calculate mean and covariance
        mean = np.mean(class_samples, axis=0)
        cov = np.cov(class_samples.T)
        
        # Generate synthetic samples
        n_synthetic_per_class = n_synthetic // len(np.unique(y))
        synthetic = np.random.multivariate_normal(mean, cov, n_synthetic_per_class)
        
        # Transform back to original scale
        synthetic = scaler.inverse_transform(synthetic)
        
        synthetic_data.append(synthetic)
        synthetic_labels.extend([class_label] * n_synthetic_per_class)
    
    # Combine all synthetic data
    X_synthetic = np.vstack(synthetic_data)
    y_synthetic = np.array(synthetic_labels)
    
    return X_synthetic, y_synthetic

def download_iris_dataset():
    """Download and save the Iris dataset with synthetic data."""
    # Load the iris dataset
    iris = load_iris()
    
    # Generate synthetic data
    X_synthetic, y_synthetic = generate_synthetic_data(
        iris.data, 
        iris.target,
        n_synthetic=1200  # Generate 1200 samples (400 per class)
    )
    
    # Create a DataFrame with synthetic data
    df = pd.DataFrame(data=X_synthetic, columns=iris.feature_names)
    df['target'] = y_synthetic
    
    # Create raw data directory if it doesn't exist
    ensure_directory_exists('data/raw')
    
    # Save the dataset
    output_path = 'data/raw/iris.csv'
    df.to_csv(output_path, index=False)
    
    logger.info(f"Dataset downloaded and saved to {output_path}")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Features: {', '.join(iris.feature_names)}")
    logger.info(f"Number of classes: {len(iris.target_names)}")
    logger.info(f"Classes: {', '.join(iris.target_names)}")
    logger.info(f"Samples per class:\n{pd.Series(y_synthetic).value_counts()}")

if __name__ == "__main__":
    download_iris_dataset() 