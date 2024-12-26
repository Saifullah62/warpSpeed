import os
import shutil
import json
import hashlib
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetBackup:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.backup_dir = self.base_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.metadata_dir = self.base_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def create_dataset_manifest(self) -> dict:
        """Create a manifest of all files in the dataset."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest = {
            "timestamp": timestamp,
            "files": [],
            "total_files": 0,
            "total_size": 0
        }

        data_dir = self.base_dir / "data"
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return manifest

        for pdf_file in data_dir.glob("**/*.pdf"):
            category = pdf_file.parent.name if pdf_file.parent.name != "data" else "uncategorized"
            file_info = {
                "path": str(pdf_file.relative_to(self.base_dir)),
                "size": pdf_file.stat().st_size,
                "hash": self.calculate_file_hash(str(pdf_file)),
                "category": category,
                "last_modified": datetime.fromtimestamp(pdf_file.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
            }
            manifest["files"].append(file_info)
            manifest["total_files"] += 1
            manifest["total_size"] += file_info["size"]

        return manifest

    def save_manifest(self, manifest: dict):
        """Save the manifest to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path = self.metadata_dir / f"dataset_manifest_{timestamp}.json"
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Manifest saved to {manifest_path}")

    def create_backup(self):
        """Create a backup of the dataset."""
        try:
            # Create manifest first
            logger.info("Creating dataset manifest...")
            manifest = self.create_dataset_manifest()
            self.save_manifest(manifest)

            # Create backup timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"dataset_backup_{timestamp}"

            # Create backup of metadata and data
            logger.info("Backing up data and metadata...")
            
            # Backup metadata
            metadata_backup = backup_path / "metadata"
            metadata_backup.mkdir(parents=True, exist_ok=True)
            if self.metadata_dir.exists():
                for metadata_file in self.metadata_dir.glob("*.json"):
                    shutil.copy2(metadata_file, metadata_backup)

            # Backup processed data
            processed_backup = backup_path / "processed_data"
            processed_backup.mkdir(parents=True, exist_ok=True)
            processed_data_dir = self.base_dir / "processed_data"
            if processed_data_dir.exists():
                for item in processed_data_dir.glob("**/*"):
                    if item.is_file():
                        relative_path = item.relative_to(processed_data_dir)
                        target_path = processed_backup / relative_path
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, target_path)

            # Create backup reference file
            reference_file = backup_path / "backup_reference.json"
            manifest_timestamp = manifest["timestamp"]  # This is already in the correct format
            reference_data = {
                "timestamp": timestamp,
                "manifest_file": manifest_timestamp,
                "total_files": manifest["total_files"],
                "total_size": manifest["total_size"],
                "backup_contents": {
                    "metadata_files": len(list(metadata_backup.glob("**/*"))),
                    "processed_files": len(list(processed_backup.glob("**/*")))
                }
            }
            
            with open(reference_file, 'w', encoding='utf-8') as f:
                json.dump(reference_data, f, indent=2)

            logger.info(f"Backup completed: {backup_path}")
            
            # Verify the backup immediately
            if self.verify_backup(timestamp):
                logger.info("Backup verification successful")
            else:
                logger.warning("Backup verification failed - some files may be missing or corrupted")
            
            return True

        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            return False

    def verify_backup(self, backup_timestamp: str) -> bool:
        """Verify the integrity of a backup."""
        try:
            backup_path = self.backup_dir / f"dataset_backup_{backup_timestamp}"
            if not backup_path.exists():
                logger.error(f"Backup directory not found: {backup_path}")
                return False

            reference_file = backup_path / "backup_reference.json"
            if not reference_file.exists():
                logger.error(f"Reference file not found: {reference_file}")
                return False
            
            with open(reference_file, 'r', encoding='utf-8') as f:
                reference_data = json.load(f)
            
            manifest_file = self.metadata_dir / f"dataset_manifest_{reference_data['manifest_file']}.json"
            if not manifest_file.exists():
                manifest_file = self.metadata_dir / f"dataset_manifest_{backup_timestamp}.json"
            
            if not manifest_file.exists():
                logger.error(f"Manifest file not found: {manifest_file}")
                return False
            
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            # Verify all files in manifest still exist and match hashes
            success = True
            for file_info in manifest["files"]:
                file_path = self.base_dir / file_info["path"]
                if not file_path.exists():
                    logger.error(f"Missing file: {file_path}")
                    success = False
                    continue
                
                current_hash = self.calculate_file_hash(str(file_path))
                if current_hash != file_info["hash"]:
                    logger.error(f"Hash mismatch for file: {file_path}")
                    success = False

            if success:
                logger.info(f"Backup verification successful for {backup_timestamp}")
            return success

        except Exception as e:
            logger.error(f"Backup verification failed: {str(e)}")
            return False

def main():
    base_dir = "c:/Users/bryan/BRYANDEVELOPMENT/STAR TREK TECH"
    backup_manager = DatasetBackup(base_dir)
    
    logger.info("Starting dataset backup...")
    if backup_manager.create_backup():
        # Get the timestamp of the backup we just created
        latest_backup = max((f.name for f in backup_manager.backup_dir.iterdir() if f.is_dir()), 
                          key=lambda x: x.split('_')[2])
        timestamp = latest_backup.split('dataset_backup_')[1]
        
        # Verify the backup
        logger.info("Verifying backup...")
        backup_manager.verify_backup(timestamp)
    
    logger.info("Backup process completed!")

if __name__ == "__main__":
    main()
