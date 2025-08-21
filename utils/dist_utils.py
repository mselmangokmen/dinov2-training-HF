import os
from accelerate import Accelerator

from typing import Set, Type

import sys
import socket
import functools 
import torch  
from accelerate.utils import DistributedDataParallelKwargs 
from utils.logger_utils import write_to_main_log
from torch import distributed as dist
 


def initialize_ddp_accelerator_from_config(config) -> Accelerator:
    # ===== CRITICAL FIX: Manual distributed initialization =====
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    
    # Set device FIRST
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Initialize distributed BEFORE Accelerator with explicit device_id
    if not dist.is_initialized():
        is_main_early = rank == 0
        if is_main_early:
            print(f"Manually initializing distributed on device cuda:{local_rank}")
            print(f"Rank {rank}/{world_size}, Local rank: {local_rank}")
        
        # Key fix: Initialize with device_id parameter
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            device_id=device  # This eliminates the warning!
        )
        
        if is_main_early:
            print("✅ Distributed initialized successfully")
    
    # Set config-based parameters
    freeze_backbone_layers = 0
    mixed_precision = 'bf16'

    if hasattr(config, 'distribution'):
        mixed_precision = config.distribution.mixed_precision
        # Set ACCELERATE_DOWNCAST_BF16 from config
        desired_downcast_bf16 = getattr(config.distribution, 'downcast_bf16', 'no').lower()
        os.environ['ACCELERATE_DOWNCAST_BF16'] = desired_downcast_bf16

    if hasattr(config.train, 'freeze_backbone_layers'):
        freeze_backbone_layers = config.train.freeze_backbone_layers
        find_unused_parameters = (freeze_backbone_layers <= 0)
    else: 
        find_unused_parameters = False
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)
    
    accelerator_kwargs = {
        "mixed_precision": mixed_precision,
        "kwargs_handlers": [ddp_kwargs], 
    }
    
    # Now Accelerator won't reinitialize distributed (already done)
    accelerator = Accelerator(**accelerator_kwargs)

    # Print statement happens only on main process *after* Accelerator init
    if accelerator.is_main_process:
        print("✅ Accelerator initialized for DDP.")
        print(f"Device: {accelerator.device}, Process: {accelerator.process_index}/{accelerator.num_processes}")

    return accelerator
 

def print_cluster_info(accelerator): 
    hostname = socket.gethostname() 
    world_size = accelerator.num_processes
 
    node_info = {
        "hostname": hostname,
        "rank": accelerator.process_index,
        "device": str(accelerator.device),
        "local_rank": int(os.environ.get('LOCAL_RANK', '0'))
    }

    if dist.is_initialized():
        node_info_list = [None] * world_size
        if accelerator.is_main_process:  
            write_to_main_log(accelerator=accelerator, result=f"Collecting basic node information from all {world_size} processes...")
        
        try:
            # Use barrier for synchronization before all_gather
            dist.barrier()
            dist.all_gather_object(node_info_list, node_info)
            
            if accelerator.is_main_process:
                write_to_main_log(accelerator=accelerator, result="✅ Node information collection completed successfully!")
                
        except Exception as e:
            if accelerator.is_main_process:
                write_to_main_log(accelerator=accelerator, result=f"⚠️ Error in all_gather: {e}")
            node_info_list = [node_info]  # Fallback
    else:
        node_info_list = [node_info]  
 
    if accelerator.is_main_process:
        nodes = {}
        for info in node_info_list:
            if info and info["hostname"] not in nodes:
                nodes[info["hostname"]] = []
            if info:
                nodes[info["hostname"]].append(info)

        output = "=" * 20 + " CLUSTER INFO SUMMARY " + "=" * 20
        write_to_main_log(accelerator=accelerator, result=output)   
        
        for hostname, gpu_list in sorted(nodes.items()):
            devices = [f"cuda:{info.get('local_rank', '?')}" for info in gpu_list]
            output = f"NODE: {hostname} has {len(gpu_list)} GPUs: {devices}"
            write_to_main_log(accelerator=accelerator, result=output)   

        output = f"TOTAL GPUs IN CLUSTER: {world_size}" 
        write_to_main_log(accelerator=accelerator, result=output)