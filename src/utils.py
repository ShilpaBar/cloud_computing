"""
Utility functions for the cloud-based machine learning system.
"""
import os
import logging
from datetime import datetime

def setup_logging(name):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(name)

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def validate_environment():
    """Validate required environment variables."""
    required_vars = [
        'GOOGLE_APPLICATION_CREDENTIALS',
        'GOOGLE_CLOUD_PROJECT',
        'GOOGLE_CLOUD_BUCKET',
        'GOOGLE_CLOUD_REGION'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
    return True

def format_duration(seconds):
    """Format duration in seconds to human-readable format."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{int(hours)}h")
    if minutes > 0:
        parts.append(f"{int(minutes)}m")
    if seconds > 0 or not parts:
        parts.append(f"{int(seconds)}s")
    
    return " ".join(parts)

# Create logs directory
ensure_directory_exists('logs')