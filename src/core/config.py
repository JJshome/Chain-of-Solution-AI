"""Configuration manager for Chain of Solution framework."""

import os
import json
import logging
from typing import Dict, Any, Optional


class CoSConfig:
    """Configuration manager for Chain of Solution framework.
    
    This class handles loading, storing, and updating configuration settings
    for the Chain of Solution framework.
    """
    
    def __init__(self, config_path=None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.logger = logging.getLogger('cos_framework.config')
        
        # Default configuration
        self.config = {
            'framework': {
                'version': '0.1.0',
                'log_level': 'INFO'
            },
            'triz': {
                'enabled': True,
                'use_extended_principles': True
            },
            'multimodal': {
                'enabled': True,
                'max_modalities': 5
            },
            'llm': {
                'enabled': True,
                'model': 'Llama3.1-70B',
                'temperature': 0.7,
                'max_tokens': 2048
            },
            'applications': {
                'cellstyle': {
                    'enabled': True,
                    'model_path': 'models/cellstyle.pt'
                },
                'soundpose': {
                    'enabled': True,
                    'model_path': 'models/soundpose.pt'
                },
                'image_enhancement': {
                    'enabled': True,
                    'model_path': 'models/image_enhancement.pt'
                }
            }
        }
        
        # Load configuration from file if provided
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """Load configuration from a JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(config_path):
            self.logger.warning(f"Configuration file not found: {config_path}")
            return False
        
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self._update_dict(self.config, loaded_config)
                self.logger.info(f"Configuration loaded from {config_path}")
                return True
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return False
    
    def _update_dict(self, d, u):
        """Recursively update a dictionary with another dictionary.
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def get(self, key_path, default=None):
        """Get a configuration value using a dot-separated path.
        
        Args:
            key_path: Dot-separated path to configuration value
            default: Default value to return if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path, value):
        """Set a configuration value using a dot-separated path.
        
        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        keys = key_path.split('.')
        config = self.config
        try:
            # Navigate to the last container in the path
            for key in keys[:-1]:
                if key not in config or not isinstance(config[key], dict):
                    config[key] = {}
                config = config[key]
            
            # Set the value
            config[keys[-1]] = value
            return True
        except Exception as e:
            self.logger.error(f"Error setting configuration value: {e}")
            return False
    
    def save_config(self, config_path):
        """Save configuration to a JSON file.
        
        Args:
            config_path: Path to save configuration file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Write config to file
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
                
            self.logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False