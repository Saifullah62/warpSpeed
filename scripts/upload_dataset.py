"""
Script to append new data to the existing Hugging Face dataset with optimized rate limiting for faster uploads.
"""
import os
import json
import time
from pathlib import Path
from huggingface_hub import HfApi, login, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
import logging
from datetime import datetime
import pickle

# Setup logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('upload_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetUploader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.api = HfApi()
        self.token = "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"
        self.repo_id = "GotThatData/warp-speed"
        self.progress_file = self.data_dir.parent / "upload_progress.pkl"
        self.uploaded_files = self._load_progress()
        
        # Optimized rate limiting settings
        self.initial_wait_time = 5  # Reduced from 10 to 5 seconds initial wait
        self.rate_limit_delay = 1  # Reduced from 2 to 1 second between uploads
        self.retry_delay = 60 * 5  # Keep 5 minutes retry delay for safety
        self.max_retries = 5
        self.batch_size = 100  # Keep batch size at 100 files
        self.batch_break_time = 15  # Reduced from 30 to 15 seconds break between batches
        self.consecutive_success = 0
        self.min_delay = 0.5  # Reduced from 1 to 0.5 second minimum delay
        self.max_consecutive = 20  # Keep at 20 for stability

    def _load_progress(self) -> set:
        """Load the set of already uploaded files."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")
        return set()

    def _save_progress(self):
        """Save the current progress."""
        with open(self.progress_file, 'wb') as f:
            pickle.dump(self.uploaded_files, f)

    def _check_repo_access(self):
        """Check if we can access the repository."""
        try:
            logger.info("Checking repository access...")
            self.api.list_repo_files(self.repo_id, repo_type="dataset")
            logger.info("Repository access confirmed.")
            return True
        except Exception as e:
            logger.error(f"Cannot access repository: {e}")
            return False

    def _handle_rate_limit(self, retry_count: int):
        """Handle rate limiting with exponential backoff."""
        if retry_count >= self.max_retries:
            raise Exception("Max retries exceeded")
        
        # Reset consecutive success counter and increase delay
        self.consecutive_success = 0
        self.rate_limit_delay = min(60, self.rate_limit_delay * 1.5)
        
        wait_time = self.retry_delay * (2 ** retry_count)
        logger.info(f"Rate limit hit. Waiting {wait_time/3600:.1f} hours before retry {retry_count + 1}/{self.max_retries}")
        logger.info(f"Increased delay between uploads to {self.rate_limit_delay} seconds")
        time.sleep(wait_time)

    def _adjust_delay(self):
        """Dynamically adjust delay based on success rate."""
        if self.consecutive_success >= self.max_consecutive:
            # Decrease delay if we've had many successful uploads
            new_delay = max(self.min_delay, self.rate_limit_delay - 1)
            if new_delay != self.rate_limit_delay:
                self.rate_limit_delay = new_delay
                logger.info(f"Decreased delay to {self.rate_limit_delay} seconds due to consistent success")
            self.consecutive_success = 0
        
    def upload_file_with_retry(self, file_path: Path, relative_path: Path):
        """Upload a single file with retry logic."""
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                logger.info(f"Attempting to upload {relative_path}")
                self.api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=str(relative_path),
                    repo_id=self.repo_id,
                    repo_type="dataset"
                )
                self.uploaded_files.add(str(relative_path))
                self._save_progress()
                logger.info(f"Successfully uploaded {relative_path}")
                
                # Increase consecutive success counter and adjust delay
                self.consecutive_success += 1
                self._adjust_delay()
                
                logger.info(f"Waiting {self.rate_limit_delay} seconds before next upload...")
                time.sleep(self.rate_limit_delay)
                return True
            except HfHubHTTPError as e:
                if "429" in str(e):  # Rate limit error
                    logger.warning(f"Rate limit reached during upload of {relative_path}")
                    self._handle_rate_limit(retry_count)
                    retry_count += 1
                else:
                    logger.error(f"Error uploading {relative_path}: {e}")
                    return False
            except Exception as e:
                logger.error(f"Unexpected error uploading {relative_path}: {e}")
                return False
        return False

    def upload_dataset(self):
        """Upload the dataset with progress tracking and dynamic rate limiting."""
        try:
            # Login to Hugging Face
            logger.info("Logging in to Hugging Face...")
            login(token=self.token)
            
            # Initial wait period
            logger.info(f"Starting with {self.initial_wait_time/60:.1f} minute wait to ensure no rate limiting...")
            time.sleep(self.initial_wait_time)
            
            # Check repository access
            if not self._check_repo_access():
                logger.error("Cannot proceed without repository access")
                return
            
            # Get total number of files
            all_files = list(self.data_dir.rglob("*"))
            files_to_upload = [f for f in all_files if f.is_file()]
            total_files = len(files_to_upload)
            uploaded_count = len(self.uploaded_files)
            
            logger.info(f"Found {total_files} files. {uploaded_count} already uploaded.")
            logger.info(f"Using optimized rate limiting:")
            logger.info(f"- Starting with {self.rate_limit_delay}s between files")
            logger.info(f"- {self.batch_break_time/60:.1f}m break every {self.batch_size} files")
            logger.info(f"- Dynamic delay adjustment based on success rate")
            
            # Upload files in batches
            batch_count = 0
            for file_path in files_to_upload:
                relative_path = file_path.relative_to(self.data_dir)
                
                # Skip if already uploaded
                if str(relative_path) in self.uploaded_files:
                    continue
                
                # Take a break between batches
                if batch_count >= self.batch_size:
                    logger.info(f"Taking a {self.batch_break_time/60:.1f} minute break between batches...")
                    time.sleep(self.batch_break_time)
                    batch_count = 0
                
                success = self.upload_file_with_retry(file_path, relative_path)
                
                if success:
                    uploaded_count += 1
                    batch_count += 1
                    logger.info(f"Progress: {uploaded_count}/{total_files} files uploaded")
                else:
                    logger.error(f"Failed to upload {relative_path} after all retries")
            
            logger.info("Upload session completed!")
            logger.info(f"Total files uploaded this session: {uploaded_count}")
            logger.info(f"View your dataset at: https://huggingface.co/datasets/{self.repo_id}")
            
        except Exception as e:
            logger.error(f"Error during upload: {e}")
            raise

def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    
    uploader = DatasetUploader(str(data_dir))
    uploader.upload_dataset()

if __name__ == "__main__":
    main()
