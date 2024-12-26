import numpy as np
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataValidationConfig:
    min_sample_size: int = 100
    max_missing_ratio: float = 0.1
    outlier_std_threshold: float = 3.0
    correlation_threshold: float = 0.95

class DataPipelineValidator:
    def __init__(self, config: Optional[DataValidationConfig] = None):
        self.config = config or DataValidationConfig()
        
    def validate_input_data(self, data: Dict[str, Any]) -> bool:
        """Validate input data against defined constraints."""
        try:
            # Check data completeness
            if not self._check_completeness(data):
                return False
                
            # Check for statistical validity
            if not self._check_statistical_validity(data):
                return False
                
            # Check for physical constraints
            if not self._check_physical_constraints(data):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
            return False
            
    def _check_completeness(self, data: Dict[str, Any]) -> bool:
        """Check if data meets completeness requirements."""
        required_fields = {'field_data', 'parameters', 'metadata'}
        
        if not all(field in data for field in required_fields):
            logger.error(f"Missing required fields: {required_fields - set(data.keys())}")
            return False
            
        if len(data['field_data']) < self.config.min_sample_size:
            logger.error(f"Insufficient data points: {len(data['field_data'])} < {self.config.min_sample_size}")
            return False
            
        return True
        
    def _check_statistical_validity(self, data: Dict[str, Any]) -> bool:
        """Validate statistical properties of the data."""
        try:
            field_data = np.array(data['field_data'])
            
            # Check for NaN/inf values
            if np.any(~np.isfinite(field_data)):
                logger.error("Data contains NaN or infinite values")
                return False
                
            # Check for outliers
            z_scores = np.abs((field_data - np.mean(field_data)) / np.std(field_data))
            if np.any(z_scores > self.config.outlier_std_threshold):
                logger.warning(f"Outliers detected: {np.sum(z_scores > self.config.outlier_std_threshold)} points")
                
            # Check for autocorrelation
            if field_data.ndim == 2:
                correlation_matrix = np.corrcoef(field_data)
                if np.any(np.abs(correlation_matrix) > self.config.correlation_threshold):
                    logger.warning("High correlation detected in field data")
                    
            return True
            
        except Exception as e:
            logger.error(f"Statistical validation error: {str(e)}")
            return False
            
    def _check_physical_constraints(self, data: Dict[str, Any]) -> bool:
        """Validate physical constraints of the data."""
        try:
            parameters = data['parameters']
            
            # Check for positive energy density
            if np.any(data['field_data'] < 0):
                logger.error("Negative energy density detected")
                return False
                
            # Check for causality preservation
            if 'c' in parameters and np.any(data['field_data'] > parameters['c']):
                logger.error("Field values exceed speed of light")
                return False
                
            # Check for quantum uncertainty principle
            if 'h_bar' in parameters:
                uncertainty = np.std(data['field_data'])
                if uncertainty < parameters['h_bar']:
                    logger.error("Field uncertainty violates quantum uncertainty principle")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Physical constraint validation error: {str(e)}")
            return False
