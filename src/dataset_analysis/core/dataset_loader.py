from typing import Dict, Any, Optional, Union
from datasets import load_dataset, Dataset, DatasetDict
import logging
from pathlib import Path
import yaml

class DatasetLoader:
    """Handles loading and initial validation of datasets from Hugging Face."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path) if config_path else {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            return {}
            
    def load_dataset(self, 
                    dataset_name: str,
                    subset: Optional[str] = None,
                    split: Optional[str] = None,
                    **kwargs) -> Union[Dataset, DatasetDict]:
        """
        Load a dataset from Hugging Face hub.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face
            subset: Specific subset of the dataset
            split: Specific split of the dataset
            **kwargs: Additional arguments for load_dataset
            
        Returns:
            Loaded dataset or dataset dictionary
        """
        try:
            self.logger.info(f"Loading dataset: {dataset_name}")
            dataset = load_dataset(dataset_name, subset, split=split, **kwargs)
            self.logger.info(f"Successfully loaded dataset: {dataset_name}")
            return dataset
        except Exception as e:
            self.logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            raise
            
    def validate_schema(self, 
                       dataset: Union[Dataset, DatasetDict],
                       schema: Dict[str, Any]) -> bool:
        """
        Validate dataset against a predefined schema.
        
        Args:
            dataset: Dataset to validate
            schema: Schema dictionary defining expected structure
            
        Returns:
            bool: True if validation passes
        """
        try:
            if isinstance(dataset, DatasetDict):
                return all(self._validate_split(split, schema) 
                         for split in dataset.values())
            return self._validate_split(dataset, schema)
        except Exception as e:
            self.logger.error(f"Schema validation error: {str(e)}")
            return False
            
    def _validate_split(self, 
                       dataset: Dataset,
                       schema: Dict[str, Any]) -> bool:
        """Validate a single dataset split against schema."""
        # Check required fields
        for field, field_type in schema.get('required_fields', {}).items():
            if field not in dataset.features:
                self.logger.error(f"Missing required field: {field}")
                return False
            if str(dataset.features[field]) != field_type:
                self.logger.error(
                    f"Invalid type for {field}: expected {field_type}, "
                    f"got {dataset.features[field]}"
                )
                return False
                
        # Check data types
        for field, field_type in schema.get('field_types', {}).items():
            if field in dataset.features:
                if str(dataset.features[field]) != field_type:
                    self.logger.error(
                        f"Invalid type for {field}: expected {field_type}, "
                        f"got {dataset.features[field]}"
                    )
                    return False
                    
        return True
