import os
from google.cloud import storage
from google.cloud import aiplatform
from dotenv import load_dotenv
import json

def test_cloud_storage():
    """Test Google Cloud Storage connection and operations."""
    try:
        # Initialize client
        storage_client = storage.Client()
        
        # Get the bucket
        bucket_name = os.getenv('GOOGLE_CLOUD_BUCKET')
        bucket = storage_client.bucket(bucket_name)
        
        # Create a test file
        test_content = "Cloud Storage Test"
        blob = bucket.blob("test.txt")
        blob.upload_from_string(test_content)
        
        # Read back the content
        downloaded_content = blob.download_as_string().decode('utf-8')
        assert downloaded_content == test_content
        
        # Clean up
        blob.delete()
        
        print("✓ Cloud Storage Test: SUCCESS")
        print(f"  - Successfully accessed bucket: {bucket_name}")
        print("  - Successfully uploaded and downloaded test file")
        print("  - Successfully deleted test file")
        return True
    except Exception as e:
        print("✗ Cloud Storage Test: FAILED")
        print(f"  Error: {str(e)}")
        return False

def test_vertex_ai():
    """Test Vertex AI connection and operations."""
    try:
        # Initialize Vertex AI
        aiplatform.init(
            project=os.getenv('GOOGLE_CLOUD_PROJECT'),
            location=os.getenv('GOOGLE_CLOUD_REGION')
        )
        
        # List available models
        models = aiplatform.Model.list()
        
        # List training pipelines
        pipelines = aiplatform.AutoMLTabularTrainingJob.list()
        
        print("✓ Vertex AI Test: SUCCESS")
        print(f"  - Project: {os.getenv('GOOGLE_CLOUD_PROJECT')}")
        print(f"  - Region: {os.getenv('GOOGLE_CLOUD_REGION')}")
        print(f"  - Number of models: {len(models)}")
        print(f"  - Number of training pipelines: {len(pipelines)}")
        return True
    except Exception as e:
        print("✗ Vertex AI Test: FAILED")
        print(f"  Error: {str(e)}")
        return False

def test_permissions():
    """Test if we have all required permissions."""
    try:
        # Get credentials info
        storage_client = storage.Client()
        credentials_info = storage_client._credentials.signer_email
        
        print("✓ Permissions Test: SUCCESS")
        print(f"  - Authenticated as: {credentials_info}")
        
        # Try to list buckets to verify permissions
        buckets = list(storage_client.list_buckets())
        print(f"  - Has access to {len(buckets)} bucket(s)")
        return True
    except Exception as e:
        print("✗ Permissions Test: FAILED")
        print(f"  Error: {str(e)}")
        return False

def main():
    print("\nTesting Cloud Components...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Verify environment variables
    required_vars = [
        'GOOGLE_APPLICATION_CREDENTIALS',
        'GOOGLE_CLOUD_PROJECT',
        'GOOGLE_CLOUD_BUCKET',
        'GOOGLE_CLOUD_REGION'
    ]
    
    print("\nChecking Environment Variables:")
    all_vars_present = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✓ {var} is set")
        else:
            print(f"✗ {var} is not set")
            all_vars_present = False
    
    if not all_vars_present:
        print("\nError: Some required environment variables are missing!")
        return
    
    print("\nTesting Cloud Storage:")
    storage_ok = test_cloud_storage()
    
    print("\nTesting Vertex AI:")
    vertex_ok = test_vertex_ai()
    
    print("\nTesting Permissions:")
    permissions_ok = test_permissions()
    
    print("\nSummary:")
    print("=" * 50)
    print(f"Cloud Storage: {'✓' if storage_ok else '✗'}")
    print(f"Vertex AI: {'✓' if vertex_ok else '✗'}")
    print(f"Permissions: {'✓' if permissions_ok else '✗'}")
    
    if storage_ok and vertex_ok and permissions_ok:
        print("\nAll cloud components are working correctly! ✓")
    else:
        print("\nSome components failed. Please check the errors above. ✗")

if __name__ == "__main__":
    main() 