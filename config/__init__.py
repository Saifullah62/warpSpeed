"""Configuration management for Star Trek Technology project."""

import os
import yaml
from typing import Dict, Any
from pathlib import Path

class ConfigManager:
    """Manages configuration loading and environment-specific settings."""

    def __init__(self, env: str = None):
        """Initialize configuration manager.
        
        Args:
            env: Environment name ('development', 'production', 'testing')
        """
        self.env = env or os.getenv('APP_ENV', 'development')
        self.config_dir = Path(__file__).parent
        self.config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> None:
        """Load configuration based on environment."""
        # Load base config
        base_config = self._load_yaml('base.yaml')
        
        # Load environment-specific config
        env_config = self._load_yaml(f'{self.env}.yaml')
        
        # Merge configurations
        self.config = self._deep_merge(base_config, env_config)

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file.
        
        Args:
            filename: Name of the YAML file
            
        Returns:
            Dict containing configuration
        """
        config_path = self.config_dir / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filename}")
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries.
        
        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if isinstance(value, dict) and key in result:
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
                
        return value if value is not None else default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key (dot notation supported)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value

# Global configuration instance
config = ConfigManager()
