import os
from src.cloud_utils import (
    upload_data_to_cloud,
    train_model_on_cloud,
    deploy_model,
    predict_with_endpoint
)
import pandas as pd
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Upload training data to cloud storage
    print("Uploading training data to Cloud Storage...")
    training_data_path = "data/cloud/training_data.csv"
    dataset_uri = upload_data_to_cloud(training_data_path)
    
    # Define advanced training parameters
    training_params = {
        "target_column": "target",
        "prediction_type": "classification",
        "budget_milli_node_hours": 2000,
        "model_display_name": "iris_classifier_v4",
        "optimization_objective": "minimize-log-loss",  # Fixed for multi-class classification
        "training_fraction_split": 0.8,
        "validation_fraction_split": 0.1,
        "test_fraction_split": 0.1
    }
    
    # Train Random Forest model with advanced configuration
    print("\nTraining Random Forest model...")
    rf_model = train_model_on_cloud(
        display_name="iris_classifier_v4",
        dataset_uri=dataset_uri,
        target_column="target",
        model_type="random_forest",
        training_params=training_params
    )
    
    # Deploy the model
    print("\nDeploying model...")
    endpoint = deploy_model(
        rf_model,
        machine_type="n1-standard-4"  # Using a more powerful machine for deployment
    )
    
    # Test predictions
    print("\nTesting predictions...")
    test_data = pd.read_csv("data/cloud/test_data.csv")
    # Remove target column from test data
    test_instances = test_data.drop('target', axis=1).to_dict('records')
    
    predictions = predict_with_endpoint(endpoint, test_instances[:5])
    print("\nSample predictions:")
    for instance, prediction in zip(test_instances[:5], predictions):
        print(f"Input: {instance}")
        print(f"Prediction: {prediction}\n")

if __name__ == "__main__":
    main() 