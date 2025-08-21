# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import math 
import os
import shutil
import json
import socket
import time
from types import SimpleNamespace 
from omegaconf import OmegaConf 
import torch.distributed as dist

from utils.standardization import standardize_model_config

def get_node_info():
    """Returns node information."""
    hostname = socket.gethostname()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if dist.is_initialized() else 0
    return hostname, rank, world_size, local_rank

def load_config(config_name: str):
    """Load configuration file."""
    if os.path.isfile(config_name): 
        config_path = config_name
    else:
        config_path = os.path.join('configs', f"{config_name}.yaml")
    return OmegaConf.load(config_path)

def write_config(cfg, output_dir, name="config.yaml"):
    """Save configuration file."""
    saved_cfg_path = os.path.join(output_dir, name)
    
    # Short wait to prevent potential writing from another process
    if dist.is_initialized() and dist.get_rank() == 0:
        time.sleep(0.5)
    
    # If file already exists, return without rewriting
    if os.path.exists(saved_cfg_path):
        hostname, rank, _, _ = get_node_info()
        
        #is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        #if is_main_process:
        #    print(f"Node: {hostname}, Rank: {rank} - Config exists: {saved_cfg_path}")
        return saved_cfg_path
    
    try:
        # Write config file
        with open(saved_cfg_path, "w") as f:
            OmegaConf.save(config=cfg, f=f)
        hostname, rank, _, _ = get_node_info()
        print(f"Node: {hostname}, Rank: {rank} - Config written: {saved_cfg_path}")
    except Exception as e:
        hostname, rank, _, _ = get_node_info()
        print(f"Node: {hostname}, Rank: {rank} - Error writing config: {e}")
    
    return saved_cfg_path

def get_default_configs(resume, config_file='ssl_default_config'):
    """Load or create default config."""
    try:
        dinov2_default_config = load_config(config_file)
        default_cfg = OmegaConf.create(dinov2_default_config)
        
        # Make paths absolute
        output_dir = os.path.abspath(default_cfg.output_folders.main_output)
        default_cfg.output_folders.main_output = output_dir
        
        model_name = default_cfg.train.model_name 
        model_path = os.path.join(output_dir, model_name)
        
        hostname, rank, _, _ = get_node_info()
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        
        if is_main_process:
            if os.path.exists(model_path) and resume:
                config_file_path = os.path.join(model_path, 'config.yaml')
                if os.path.exists(config_file_path):
                    try:
                        resume_conf = OmegaConf.load(config_file_path)
                        default_cfg = OmegaConf.create(resume_conf)
                        # Ensure path is still absolute
                        default_cfg.output_folders.main_output = os.path.abspath(default_cfg.output_folders.main_output)
                        #print(f"Node: {hostname}, Rank: {rank} - Resumed config from {config_file_path}")
                    except Exception as e:
                        print(f"Node: {hostname}, Rank: {rank} - Error loading config: {e}")
            elif os.path.exists(model_path) and not resume:
                try:
                    # Safe deletion in multi-node
                    if os.path.exists(model_path):

                        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
                        if is_main_process:
                            print(f"Node: {hostname}, Rank: {rank} - Removing existing model path: {model_path}")
                        shutil.rmtree(model_path)
                except Exception as e:
                    print(f"Node: {hostname}, Rank: {rank} - Error removing model path: {e}")
        
        return default_cfg
    except Exception as e:
        hostname, rank, _, _ = get_node_info()
        print(f"Node: {hostname}, Rank: {rank} - Error in get_default_configs: {e}")
        raise

def create_output_dirs(cfg):
    """Create output directories automatically based on cfg.output_folders contents."""
    hostname, rank, *_ = get_node_info()
    
    try:
        output_dir = os.path.abspath(cfg.output_folders.main_output)
        main_model_dir = os.path.join(output_dir, cfg.train.model_name)
        
        # Ana dizin + cfg.output_folders içindeki tüm string değerler
        dirs_to_create = [main_model_dir]
        for attr_name in dir(cfg.output_folders):
            if not attr_name.startswith('_') and attr_name != 'main_output':
                attr_value = getattr(cfg.output_folders, attr_name, None)
                if isinstance(attr_value, str):
                    dirs_to_create.append(os.path.join(main_model_dir, attr_value))
        
        # Dizinleri oluştur
        for directory in dirs_to_create:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"Node: {hostname}, Rank: {rank} - Created: {directory}")
        
        time.sleep(0.5)
        write_config(cfg, main_model_dir)
        
    except Exception as e:
        print(f"Node: {hostname}, Rank: {rank} - Error: {e}")
        raise
def setup(config_file='ssl_default_config', resume=True):
    """Main setup function with proper synchronization."""
    hostname, rank, world_size, local_rank = get_node_info() 
    
    # Initial barrier for synchronization
    if dist.is_initialized():
        dist.barrier()
    
    # Get the configuration
    cfg = get_default_configs(resume, config_file=config_file)
    
    # Absolute paths
    output_dir = os.path.abspath(cfg.output_folders.main_output)
    cfg.output_folders.main_output = output_dir
    
    # Only have rank 0 create directories and config file
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    
    if is_main_process:
        create_output_dirs(cfg)
        print(f"Node: {hostname}, Rank: {rank} - Created directories and config at {output_dir}")
    
    # Make sure all processes see the directories/files created by rank 0
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception as e:
            print(f"Node: {hostname}, Rank: {rank} - Error in barrier: {e}")
     
    return cfg

def load_model_configs(model_type, config_path="configs/models/models.json"): 
    """Load model configuration."""
    try:
        with open(config_path, 'r') as f:
            all_configs = json.load(f, object_hook=lambda d: SimpleNamespace(**d)) 
            
            configs = getattr(all_configs, model_type, None)
            if configs: 
                configs = standardize_model_config(configs)
        return configs
    except FileNotFoundError:
        hostname, rank, _, _ = get_node_info()
        print(f"Node: {hostname}, Rank: {rank} - Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        hostname, rank, _, _ = get_node_info()
        print(f"Node: {hostname}, Rank: {rank} - Configuration file is not a valid JSON: {config_path}")
        raise ValueError(f"Configuration file is not a valid JSON: {config_path}")
 