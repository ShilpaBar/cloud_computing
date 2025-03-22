import os
from google.cloud import aiplatform
from dotenv import load_dotenv
import time
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
import json

def setup_email_config():
    """
    Set up email configuration for notifications.
    Returns a dictionary with email settings.
    """
    # You should set these in your .env file
    return {
        'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
        'smtp_port': int(os.getenv('SMTP_PORT', '587')),
        'sender_email': os.getenv('SENDER_EMAIL'),
        'sender_password': os.getenv('SENDER_PASSWORD'),
        'recipient_email': os.getenv('RECIPIENT_EMAIL')
    }

def send_notification(subject, message, email_config):
    """Send email notification about training status."""
    if not all([email_config['sender_email'], email_config['sender_password'], email_config['recipient_email']]):
        print("Email configuration incomplete. Skipping notification.")
        return

    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = email_config['sender_email']
    msg['To'] = email_config['recipient_email']

    try:
        with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
            server.starttls()
            server.login(email_config['sender_email'], email_config['sender_password'])
            server.send_message(msg)
        print(f"Notification sent: {subject}")
    except Exception as e:
        print(f"Failed to send notification: {str(e)}")

def monitor_training():
    """Monitor training progress and send notifications."""
    # Load environment variables
    load_dotenv()
    
    # Initialize Vertex AI
    aiplatform.init(
        project=os.getenv('GOOGLE_CLOUD_PROJECT'),
        location=os.getenv('GOOGLE_CLOUD_REGION')
    )
    
    # Set up email configuration
    email_config = setup_email_config()
    
    # Get list of training jobs
    jobs = aiplatform.AutoMLTabularTrainingJob.list()
    
    if not jobs:
        print("No training jobs found.")
        return
    
    # Get the most recent job
    job = jobs[0]
    print(f"\nMonitoring training job: {job.display_name}")
    
    # Store metrics history
    metrics_history = {}
    last_status = None
    
    while True:
        try:
            # Get current status
            status = job.status
            
            # Print status if changed
            if status != last_status:
                print(f"\nStatus: {status}")
                last_status = status
                
                # Send notification on status change
                send_notification(
                    f"ML Training Status Update: {status}",
                    f"Training job {job.display_name} status changed to {status}",
                    email_config
                )
            
            # If training is completed
            if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
                message = f"Training job {job.display_name} {status.lower()}"
                if status == 'COMPLETED':
                    # Get final metrics
                    try:
                        model = job.get_model()
                        metrics = model.get_model_evaluation()
                        message += f"\n\nFinal Metrics:\n{json.dumps(metrics, indent=2)}"
                    except Exception as e:
                        message += f"\n\nCould not retrieve metrics: {str(e)}"
                
                send_notification(
                    f"ML Training {status}: {job.display_name}",
                    message,
                    email_config
                )
                break
            
            # Get current metrics if available
            try:
                current_metrics = job.get_metrics()
                if current_metrics != metrics_history.get('latest'):
                    metrics_history['latest'] = current_metrics
                    print("\nCurrent Metrics:")
                    print(json.dumps(current_metrics, indent=2))
            except Exception as e:
                print(f"Could not retrieve metrics: {str(e)}")
            
            # Wait before next check
            time.sleep(300)  # Check every 5 minutes
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"\nError during monitoring: {str(e)}")
            time.sleep(60)  # Wait a minute before retrying

if __name__ == "__main__":
    monitor_training() 