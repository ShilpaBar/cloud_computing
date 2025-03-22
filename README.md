# Cloud-Based Machine Learning System

A robust machine learning system built on Google Cloud Platform (GCP) using Vertex AI for training and deploying ML models. This project provides a complete pipeline for training, deploying, and monitoring machine learning models in the cloud.

## Project Structure

```
cloud_based_ml/
├── data/
│   └── cloud/
│       ├── training_data.csv    # Training dataset
│       └── test_data.csv        # Test dataset
├── src/
│   ├── __init__.py
│   ├── cloud_utils.py          # Cloud operations utilities
│   └── utils.py                # General utilities
├── cloud_train_deploy.py       # Main training and deployment script
├── check_training_status.py    # Basic training status monitoring
├── test_cloud_components.py    # Cloud setup verification
├── monitor_training.py         # Advanced monitoring with notifications
├── prepare_cloud_data.py       # Data preparation utilities
├── requirements.txt           # Project dependencies
├── .env                       # Environment configuration
├── key.json                   # GCP service account key
└── README.md                  # Project documentation
```

## Prerequisites

1. Python 3.8 or higher
2. Google Cloud Platform account
3. Google Cloud SDK installed
4. Service account with necessary permissions

## Dependencies

The project requires the following main packages:

- google-cloud-storage >= 2.14.0
- google-cloud-aiplatform >= 1.38.1
- pandas >= 2.2.0
- python-dotenv >= 1.0.1
- scikit-learn >= 1.4.0
- numpy >= 1.26.0

## Setup

1. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   Create a `.env` file in the project root with:

   ```
   GOOGLE_APPLICATION_CREDENTIALS=./key.json
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_CLOUD_BUCKET=your-bucket-name
   GOOGLE_CLOUD_REGION=us-central1
   ```

3. **Service Account Setup**
   - Place your service account key file (`key.json`) in the project root
   - Required roles:
     - Vertex AI User
     - Storage Object Viewer
     - Storage Object Creator

## Workflow

1. **Prepare Your Data**

   ```bash
   python prepare_cloud_data.py
   ```

   This script helps prepare and validate your data for cloud training.

2. **Verify Cloud Setup**

   ```bash
   python test_cloud_components.py
   ```

   Ensures all cloud components are properly configured.

3. **Train and Deploy Model**

   ```bash
   python cloud_train_deploy.py
   ```

   Handles the complete pipeline from training to deployment.

4. **Monitor Training**
   ```bash
   python check_training_status.py  # Basic monitoring
   # or
   python monitor_training.py       # Advanced monitoring with notifications
   ```

## Core Components

### 1. Cloud Utilities (`src/cloud_utils.py`)

- Data upload to Cloud Storage
- Model training configuration
- Model deployment management
- Prediction serving

### 2. Data Preparation (`prepare_cloud_data.py`)

- Data validation
- Format conversion
- Feature preprocessing
- Train-test splitting

### 3. Training and Deployment (`cloud_train_deploy.py`)

- AutoML model training
- Model evaluation
- Endpoint deployment
- Test predictions

### 4. Monitoring (`monitor_training.py`)

- Real-time training status
- Performance metrics
- Resource utilization
- Email notifications

## Model Configuration

```python
training_params = {
    "target_column": "target",
    "prediction_type": "classification",
    "budget_milli_node_hours": 2000,
    "model_display_name": "model_name",
    "optimization_objective": "maximize-au-roc",
    "training_fraction_split": 0.8,
    "validation_fraction_split": 0.1,
    "test_fraction_split": 0.1
}
```

## Best Practices

1. **Data Management**

   - Validate data before upload
   - Use consistent naming conventions
   - Keep data in `data/cloud/` directory

2. **Resource Management**

   - Monitor training costs
   - Clean up unused endpoints
   - Use appropriate machine types

3. **Security**
   - Secure credentials
   - Use minimum permissions
   - Rotate service account keys

## Troubleshooting

1. **Connection Issues**

   - Check `.env` configuration
   - Verify service account permissions
   - Confirm Google Cloud SDK setup

2. **Training Failures**

   - Validate data format
   - Check resource quotas
   - Review error logs

3. **Deployment Issues**
   - Verify region availability
   - Check endpoint configuration
   - Monitor resource allocation

## Support

For assistance:

1. Check this documentation
2. Review troubleshooting guide
3. Submit an issue
4. Contact project maintainers

## License

This project is licensed under the MIT License.
