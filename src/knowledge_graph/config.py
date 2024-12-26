import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import yaml
import json
from pathlib import Path

@dataclass
class KnowledgeGraphConfig:
    """
    Centralized configuration management for Knowledge Graph components.
    
    Supports multiple configuration sources:
    1. Environment variables
    2. YAML configuration files
    3. JSON configuration files
    4. Default values
    """
    
    # Entity Extraction Configuration
    entity_extraction: Dict[str, Any] = field(default_factory=lambda: {
        'models': {
            'spacy': {
                'model_name': 'en_core_web_trf',
                'confidence_threshold': 0.7
            },
            'huggingface': {
                'model_name': 'allenai/scibert_scivocab_uncased',
                'confidence_threshold': 0.6
            }
        },
        'multi_modal': {
            'enabled': True,
            'strategies': ['textual', 'contextual', 'semantic']
        }
    })
    
    # Relationship Mapping Configuration
    relationship_mapping: Dict[str, Any] = field(default_factory=lambda: {
        'confidence_scoring': {
            'base_weight': 0.5,
            'semantic_proximity_weight': 0.3,
            'type_compatibility_weight': 0.2
        },
        'max_relationship_distance': 3,
        'pruning_threshold': 0.4
    })
    
    # Graph Construction Configuration
    graph_construction: Dict[str, Any] = field(default_factory=lambda: {
        'versioning': {
            'max_versions': 10,
            'auto_prune': True
        },
        'visualization': {
            'enabled': True,
            'output_format': ['networkx', 'graphviz', 'json']
        }
    })
    
    # Logging Configuration
    logging: Dict[str, Any] = field(default_factory=lambda: {
        'level': 'INFO',
        'console_output': True,
        'file_output': True,
        'log_dir': 'logs/knowledge_graph'
    })
    
    # Performance and Resource Configuration
    performance: Dict[str, Any] = field(default_factory=lambda: {
        'max_concurrent_tasks': 4,
        'memory_limit_mb': 2048,
        'cache_enabled': True,
        'cache_expiry_minutes': 60
    })
    
    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> 'KnowledgeGraphConfig':
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
        
        Returns:
            Configured KnowledgeGraphConfig instance
        """
        if not config_path:
            # Default config path
            config_path = os.path.join(
                os.path.dirname(__file__), 
                '..', '..', 'config', 'knowledge_graph_config.yaml'
            )
        
        # Ensure config file exists
        if not os.path.exists(config_path):
            return cls()  # Return default config
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f) or {}
            
            # Merge YAML config with default config
            return cls(**config_dict)
        
        except (yaml.YAMLError, IOError) as e:
            print(f"Error loading YAML config: {e}")
            return cls()
    
    @classmethod
    def from_json(cls, config_path: Optional[str] = None) -> 'KnowledgeGraphConfig':
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the JSON configuration file
        
        Returns:
            Configured KnowledgeGraphConfig instance
        """
        if not config_path:
            # Default config path
            config_path = os.path.join(
                os.path.dirname(__file__), 
                '..', '..', 'config', 'knowledge_graph_config.json'
            )
        
        # Ensure config file exists
        if not os.path.exists(config_path):
            return cls()  # Return default config
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Merge JSON config with default config
            return cls(**config_dict)
        
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading JSON config: {e}")
            return cls()
    
    def save_config(self, format: str = 'yaml') -> None:
        """
        Save current configuration to a file.
        
        Args:
            format: Configuration file format ('yaml' or 'json')
        """
        config_dir = Path(os.path.join(os.path.dirname(__file__), '..', '..', 'config'))
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            'entity_extraction': self.entity_extraction,
            'relationship_mapping': self.relationship_mapping,
            'graph_construction': self.graph_construction,
            'logging': self.logging,
            'performance': self.performance
        }
        
        if format == 'yaml':
            output_path = config_dir / 'knowledge_graph_config.yaml'
            with open(output_path, 'w') as f:
                yaml.safe_dump(config_data, f, default_flow_style=False)
        elif format == 'json':
            output_path = config_dir / 'knowledge_graph_config.json'
            with open(output_path, 'w') as f:
                json.dump(config_data, f, indent=4)
        else:
            raise ValueError(f"Unsupported config format: {format}")
    
    def get_config_section(self, section: str) -> Dict[str, Any]:
        """
        Retrieve a specific configuration section.
        
        Args:
            section: Configuration section name
        
        Returns:
            Configuration dictionary for the specified section
        """
        return getattr(self, section, {})

# Global configuration instance
CONFIG = KnowledgeGraphConfig()

# Utility functions for configuration management
def load_config(config_type: str = 'yaml') -> KnowledgeGraphConfig:
    """
    Load configuration based on specified type.
    
    Args:
        config_type: Configuration file type ('yaml' or 'json')
    
    Returns:
        Configured KnowledgeGraphConfig instance
    """
    if config_type == 'yaml':
        return KnowledgeGraphConfig.from_yaml()
    elif config_type == 'json':
        return KnowledgeGraphConfig.from_json()
    else:
        raise ValueError(f"Unsupported config type: {config_type}")

def get_config_value(section: str, key: str, default: Any = None) -> Any:
    """
    Retrieve a specific configuration value.
    
    Args:
        section: Configuration section name
        key: Configuration key
        default: Default value if key is not found
    
    Returns:
        Configuration value or default
    """
    try:
        return CONFIG.get_config_section(section).get(key, default)
    except Exception:
        return default

# Optional: Create default configuration files if they don't exist
def initialize_config_files():
    """
    Create default configuration files if they don't exist.
    """
    config_dir = Path(os.path.join(os.path.dirname(__file__), '..', '..', 'config'))
    config_dir.mkdir(parents=True, exist_ok=True)
    
    yaml_path = config_dir / 'knowledge_graph_config.yaml'
    json_path = config_dir / 'knowledge_graph_config.json'
    
    if not yaml_path.exists():
        CONFIG.save_config(format='yaml')
    
    if not json_path.exists():
        CONFIG.save_config(format='json')

# Initialize configuration files when module is imported
initialize_config_files()
