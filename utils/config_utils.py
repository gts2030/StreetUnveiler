"""
Configuration utilities for StreetUnveiler training.

This module contains functions for loading, merging, and managing
training configurations from YAML files.
"""

import yaml
import os


def get_default_config():
    """
    Get default configuration values.
    
    Returns:
        dict: Default configuration dictionary
    """
    return {

        'uncertainty': {
            'ssim_median_filter_size': 5,
            'ssim_window_size': 7,
            'uncer_depth_mult': 0.1,
            'opacity_th_for_uncer_loss': 0.9,
            'fallback_loss_weight': 0.01,
            'beta_epsilon': 1e-8
        },
        'loss_weights': {
            'semantic_ce_weights': [1.0, 1.0, 1.0, 1.0, 0.2, 1.0],
            'render_lambda5': 1.0,
            'render_lambda6': 1.0,
            'render_lambda7': 0.01,
            'uncertainty_lambda1': 1.0,
            'uncertainty_lambda2': 0.01,
            'uncertainty_lambda3': 0.01
        },


        'semantic': {
            'sky_class_name': 'sky',
            'vegetation_class_name': 'vegetation',
            'dont_prune_classes': ['sky', 'vegetation']
        },
        'feature_cfg': {
            'device': 'cuda',
            'mono_prior': {
                'feature_extractor': 'dinov2_vits14_reg',
                'depth': 'metric3d_vit_large'
            },
            'data': {
                'output': None  # Will be set dynamically to dataset.model_path
            },
            'scene': 'default_scene',  # Will be overridden if dataset has scene_name
            'cam': {
                'fx': 1000.0
            }
        },

    }


def merge_configs(default_config, user_config):
    """
    Deep merge user config with default config.
    
    Args:
        default_config (dict): Default configuration
        user_config (dict): User provided configuration
        
    Returns:
        dict: Merged configuration
    """
    if user_config is None:
        return default_config
    
    merged = default_config.copy()
    
    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def load_training_config(config_path="configs/training_config.yaml"):
    """
    Load training configuration from YAML file and merge with defaults.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary merged with defaults
    """
    default_config = get_default_config()
    
    try:
        # Check if config path exists
        if not os.path.exists(config_path):
            print(f"Configuration file not found: {config_path}")
            print("Using default configuration...")
            return default_config
            
        with open(config_path, 'r') as file:
            user_config = yaml.safe_load(file)
            
        # Handle empty config file
        if user_config is None:
            print(f"Configuration file is empty: {config_path}")
            print("Using default configuration...")
            return default_config
            
        print(f"Loaded training configuration from: {config_path}")
        config = merge_configs(default_config, user_config)
        return config
        
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        print("Using default configuration...")
        return default_config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration...")
        return default_config


def save_config(config, save_path):
    """
    Save configuration to YAML file.
    
    Args:
        config (dict): Configuration dictionary to save
        save_path (str): Path to save the configuration file
    """
    try:
        # Only create directory if there's a directory path
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(save_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2, sort_keys=True)
        print(f"Configuration saved to: {save_path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")


def validate_config(config):
    """
    Validate configuration dictionary to ensure all required keys are present.
    
    Args:
        config (dict): Configuration dictionary to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    default_config = get_default_config()
    
    def check_keys(default_dict, config_dict, path=""):
        for key in default_dict:
            current_path = f"{path}.{key}" if path else key
            
            if key not in config_dict:
                print(f"Missing configuration key: {current_path}")
                return False
                
            if isinstance(default_dict[key], dict):
                if not isinstance(config_dict[key], dict):
                    print(f"Configuration key should be a dictionary: {current_path}")
                    return False
                if not check_keys(default_dict[key], config_dict[key], current_path):
                    return False
                    
        return True
    
    return check_keys(default_config, config)


def print_config_summary(config):
    """
    Print a summary of the current configuration.
    
    Args:
        config (dict): Configuration dictionary to summarize
    """
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION SUMMARY")
    print("="*50)
    
    # Key settings summary
    feature_cfg = config.get('feature_cfg', {})
    uncertainty_cfg = config.get('uncertainty', {})
    loss_cfg = config.get('loss_weights', {})
    
    print(f"Depth Estimator: {feature_cfg.get('mono_prior', {}).get('depth', 'N/A')}")
    print(f"Feature Extractor: {feature_cfg.get('mono_prior', {}).get('feature_extractor', 'N/A')}")
    print(f"Device: {feature_cfg.get('device', 'N/A')}")
    print(f"Focal Length: {feature_cfg.get('cam', {}).get('fx', 'N/A')}")
    print(f"Uncertainty Lambda1: {loss_cfg.get('uncertainty_lambda1', 'N/A')}")
    print(f"Uncertainty Lambda2: {loss_cfg.get('uncertainty_lambda2', 'N/A')}")
    print(f"SSIM Window Size: {uncertainty_cfg.get('ssim_window_size', 'N/A')}")
    print("="*50 + "\n") 