import os
from google.cloud import aiplatform
from dotenv import load_dotenv
import time

def get_job_status():
    """Get the status of all training jobs."""
    try:
        # Initialize Vertex AI
        aiplatform.init(
            project=os.getenv('GOOGLE_CLOUD_PROJECT'),
            location=os.getenv('GOOGLE_CLOUD_REGION')
        )
        
        # Get all training jobs
        jobs = aiplatform.AutoMLTabularTrainingJob.list()
        
        if not jobs:
            print("No training jobs found.")
            return
        
        print("\nTraining Jobs Status:")
        print("=" * 50)
        
        for job in jobs:
            # Get detailed job information
            job_info = job.to_dict()
            
            print(f"\nJob Name: {job_info.get('display_name', 'N/A')}")
            print(f"Status: {job_info.get('state', 'N/A')}")
            print(f"Create Time: {job_info.get('create_time', 'N/A')}")
            print(f"Start Time: {job_info.get('start_time', 'N/A')}")
            print(f"End Time: {job_info.get('end_time', 'N/A') or 'Still running'}")
            
            # If job has error
            if job_info.get('error'):
                print(f"Error: {job_info['error']}")
            
            # If job has metrics
            if job_info.get('final_model_stats'):
                print("\nModel Statistics:")
                stats = job_info['final_model_stats']
                for metric, value in stats.items():
                    print(f"  {metric}: {value}")
            
            print("-" * 50)
    except Exception as e:
        print(f"Error checking job status: {str(e)}")
        return False

def main():
    print("Loading environment variables...")
    load_dotenv()
    
    print(f"Checking training jobs in project: {os.getenv('GOOGLE_CLOUD_PROJECT')}")
    print(f"Region: {os.getenv('GOOGLE_CLOUD_REGION')}")
    
    get_job_status()

if __name__ == "__main__":
    main() 