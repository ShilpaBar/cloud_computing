import os
from google.cloud import storage
from google.cloud import aiplatform
import pandas as pd

def upload_data_to_cloud(local_data_path, cloud_data_path=None):
    """
    Upload data to Google Cloud Storage.
    
    Args:
        local_data_path (str): Path to local data file
        cloud_data_path (str, optional): Path in cloud storage. If None, uses the same filename
    
    Returns:
        str: GCS URI of uploaded file
    """
    storage_client = storage.Client()
    bucket_name = os.getenv('GOOGLE_CLOUD_BUCKET')
    bucket = storage_client.bucket(bucket_name)
    
    # If no cloud path specified, use the filename from local path
    if cloud_data_path is None:
        cloud_data_path = os.path.basename(local_data_path)
    
    blob = bucket.blob(cloud_data_path)
    blob.upload_from_filename(local_data_path)
    
    gcs_uri = f"gs://{bucket_name}/{cloud_data_path}"
    print(f"Data uploaded to {gcs_uri}")
    return gcs_uri

def train_model_on_cloud(
    display_name,
    dataset_uri,
    target_column,
    model_type="random_forest",
    training_params=None
):
    """
    Train a model using Vertex AI AutoML.
    
    Args:
        display_name (str): Name for the training job
        dataset_uri (str): GCS URI of the training dataset
        target_column (str): Name of the target column
        model_type (str): Type of model to train ("random_forest" or "svm")
        training_params (dict, optional): Additional training parameters
    
    Returns:
        Model: Trained model object
    """
    aiplatform.init(
        project=os.getenv('GOOGLE_CLOUD_PROJECT'),
        location=os.getenv('GOOGLE_CLOUD_REGION')
    )
    
    # Create dataset
    dataset = aiplatform.TabularDataset.create(
        display_name=f"{display_name}_dataset",
        gcs_source=dataset_uri
    )
    
    # Define default training parameters for classification
    if training_params is None:
        training_params = {
            "target_column": target_column,
            "prediction_type": "classification",
            "budget_milli_node_hours": 1000,
            "model_display_name": f"{display_name}_model",
            "optimization_objective": "minimize-log-loss"
        }
    
    # Define column transformations for the Iris dataset features
    column_transformations = [
        {
            "numeric": {
                "column_name": "sepal length (cm)"
            }
        },
        {
            "numeric": {
                "column_name": "sepal width (cm)"
            }
        },
        {
            "numeric": {
                "column_name": "petal length (cm)"
            }
        },
        {
            "numeric": {
                "column_name": "petal width (cm)"
            }
        }
    ]
    
    # Start training job
    job = aiplatform.AutoMLTabularTrainingJob(
        display_name=display_name,
        optimization_prediction_type="classification",
        column_transformations=column_transformations,
        optimization_objective="minimize-log-loss"  # Fixed for multi-class classification
    )
    
    model = job.run(
        dataset=dataset,
        target_column=target_column,
        budget_milli_node_hours=training_params.get("budget_milli_node_hours", 1000),
        model_display_name=training_params.get("model_display_name", f"{display_name}_model"),
        training_fraction_split=training_params.get("training_fraction_split", 0.8),
        validation_fraction_split=training_params.get("validation_fraction_split", 0.1),
        test_fraction_split=training_params.get("test_fraction_split", 0.1),
        sync=True
    )
    
    return model

def deploy_model(model, machine_type="n1-standard-2"):
    """
    Deploy a trained model to an endpoint.
    
    Args:
        model: Trained model object
        machine_type (str): Type of machine to use for deployment
    
    Returns:
        Endpoint: Deployed model endpoint
    """
    endpoint = model.deploy(
        machine_type=machine_type,
        min_replica_count=1,
        max_replica_count=1
    )
    
    print(f"Model deployed to endpoint: {endpoint.resource_name}")
    return endpoint

def predict_with_endpoint(endpoint, instances):
    """
    Make predictions using a deployed model endpoint.
    
    Args:
        endpoint: Deployed model endpoint
        instances: List of instances to predict
    
    Returns:
        List of predictions
    """
    predictions = endpoint.predict(instances=instances)
    return predictions