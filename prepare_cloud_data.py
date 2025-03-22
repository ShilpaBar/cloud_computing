import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils import setup_logging, ensure_directory_exists, validate_environment
from dotenv import load_dotenv

# Set up logging
logger = setup_logging('data_preparation')

def validate_data(df):
    """
    Validate the input data for training.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        bool: True if validation passes, raises exception otherwise
    """
    # Check for null values
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Found null values in columns: {null_cols}")
    
    # Check for numeric features
    non_numeric = df.select_dtypes(exclude=['number']).columns
    if len(non_numeric) > 1:  # Excluding target column
        raise ValueError(f"Found non-numeric features: {non_numeric}")
    
    # Check for infinite values
    if np.isinf(df.select_dtypes(include=['number']).values).any():
        raise ValueError("Found infinite values in the dataset")
    
    logger.info("Data validation passed successfully")
    return True

def prepare_data(input_file, output_dir, test_size=0.2, random_state=42):
    """
    Prepare data for cloud training.
    
    Args:
        input_file (str): Path to input CSV file
        output_dir (str): Directory to save processed files
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
    """
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Validate data
    validate_data(df)
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Reconstruct training and test datasets
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Save processed datasets
    ensure_directory_exists(output_dir)
    train_path = os.path.join(output_dir, 'training_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Saved training data ({len(train_df)} rows) to {train_path}")
    logger.info(f"Saved test data ({len(test_df)} rows) to {test_path}")
    
    # Log data statistics
    logger.info("\nData Statistics:")
    logger.info(f"Number of features: {len(X.columns)}")
    logger.info(f"Training set size: {len(train_df)}")
    logger.info(f"Test set size: {len(test_df)}")
    logger.info(f"Target distribution:\n{y.value_counts(normalize=True)}")

def main():
    """Main function to prepare data for cloud training."""
    load_dotenv()
    validate_environment()
    
    # Define paths
    input_file = "data/raw/iris.csv"  # Update with your input file
    output_dir = "data/cloud"
    
    try:
        prepare_data(input_file, output_dir)
        logger.info("Data preparation completed successfully")
    except Exception as e:
        logger.error(f"Error during data preparation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 