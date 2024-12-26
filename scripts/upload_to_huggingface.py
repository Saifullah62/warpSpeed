import os
import json
from huggingface_hub import HfApi
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def upload_to_huggingface(
    repo_id="GotThatData/warp-speed",
    token=None,
    data_dir="data/nist/data/nist",
    file_patterns=["nist_constants.json"]
):
    """
    Upload data files to Hugging Face dataset repository.
    
    Args:
        repo_id (str): The Hugging Face repository ID (username/repo-name)
        token (str): Hugging Face API token. If None, will look for HUGGINGFACE_TOKEN env var
        data_dir (str): Directory containing the data files
        file_patterns (list): List of file patterns to upload
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Get token from environment if not provided
        if token is None:
            token = os.environ.get("HUGGINGFACE_TOKEN")
            if token is None:
                raise ValueError("No Hugging Face token provided. Set HUGGINGFACE_TOKEN environment variable or pass token directly.")
        
        # Initialize Hugging Face API
        api = HfApi()
        
        # Ensure the repository exists (will create if it doesn't)
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
            logger.info(f"Repository {repo_id} is ready")
        except Exception as e:
            logger.error(f"Error creating/accessing repository: {e}")
            raise

        # Upload each file
        for pattern in file_patterns:
            file_path = os.path.join(data_dir, pattern)
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            # For JSON files, validate before upload
            if file_path.endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        json.load(f)  # Validate JSON
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON file {file_path}: {e}")
                    continue

            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=f"data/{os.path.basename(file_path)}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token
                )
                logger.info(f"Successfully uploaded {file_path}")
            except Exception as e:
                logger.error(f"Error uploading {file_path}: {e}")
                raise

        logger.info(f"Dataset successfully uploaded to {repo_id}")
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise

if __name__ == "__main__":
    try:
        upload_to_huggingface()
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise
