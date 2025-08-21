import os
import shutil
import logging
import json
from typing import Optional, Tuple, Any
import torch
import torch.distributed as dist
import re

from utils.logger_utils import write_to_main_log
from accelerate import Accelerator
from accelerate.utils import DistributedType # Needed for checking init type

from safetensors.torch import save_file  , load_file

class DDPModelBackboneCheckpointManager: 
    def __init__(self,
                 model: torch.nn.Module,
                 config: Any, # Configuration object
                 accelerator: Accelerator,
                 save_dir: Optional[str] = None, # Can override default save_dir from config
                 max_to_keep: Optional[int] = 3,
                 ):
        # Check only if Accelerator reports distributed is initialized.
        if accelerator.state.distributed_type != DistributedType.NO and (not dist.is_available() or not dist.is_initialized()):
             raise RuntimeError("Distributed environment is not initialized properly by Accelerator.")

        self.model = model
        self.config = config
        self.accelerator = accelerator
        self.is_main_process = accelerator.is_main_process

   
        # Determine save directory from config if not provided
        if save_dir is None:
             # Assuming config has output_folders and train.model_name
             # Using teacher_checkpoint_dir for backbone saves
             self.save_dir = os.path.join(
                 config.output_folders.main_output,
                 config.train.model_name,
                 config.output_folders.teacher_checkpoint_dir
             )
        else:
             self.save_dir = save_dir

        self.max_to_keep = max_to_keep if max_to_keep is not None and max_to_keep > 0 else None

        # Ensure save directory exists on main process
        if self.is_main_process: # No need to check os.path.exists first before makedirs with exist_ok=True
            write_to_main_log(accelerator=self.accelerator, result=f"[Rank 0] Ensuring backbone checkpoint directory exists: {self.save_dir}")
            os.makedirs(self.save_dir, exist_ok=True)

        # Synchronize after directory creation
        self.accelerator.wait_for_everyone()
        write_to_main_log(accelerator=self.accelerator, result=f"[{self.accelerator.process_index}] ModelBackboneCheckpointManager initialized. Save directory: {self.save_dir}")


    def save(self, iteration: int): 
        if not self.is_main_process: 
            self.accelerator.wait_for_everyone()
            return

        current_ckpt_dir = os.path.join(self.save_dir, f"iter_{iteration}")
        
        try: 

            os.makedirs(current_ckpt_dir, exist_ok=True)
            if hasattr(self.model, "student_shadow"): 
                backbone = self.accelerator.unwrap_model(self.model.student_shadow.backbone)
            else: 
                if hasattr(self.model, "teacher_model"):
                    backbone = self.accelerator.unwrap_model(self.model.teacher_model.backbone)
                else: 
                    backbone = self.accelerator.unwrap_model(self.model.backbone)
                
 
            safetensors_path = os.path.join(current_ckpt_dir, "model.safetensors")
            if hasattr(backbone, "save_pretrained"):
                    backbone.save_pretrained(current_ckpt_dir, safe_serialization=True)
            else: 
                    state_dict = backbone.state_dict()
                    save_file(state_dict, safetensors_path)
            write_to_main_log(accelerator=self.accelerator, result=f"[Rank 0] Backbone state dict saved to {safetensors_path}")

            # No metadata file creation - we'll use directory names instead

            write_to_main_log(accelerator=self.accelerator, result=f"[Rank 0] Backbone checkpoint for iteration {iteration} saved successfully.")

            # Clean old checkpoints *only if max_to_keep is configured*
            if self.max_to_keep is not None:
                self._clean_checkpoints()

        except Exception as e:
            write_to_main_log(accelerator=self.accelerator, result=f"[Rank 0] Failed to save backbone checkpoint for iteration {iteration}: {e}", type='error')
            import traceback
            write_to_main_log(accelerator=self.accelerator, result=traceback.format_exc(), type='error')

            # Clean up potentially incomplete directory if save failed
            if os.path.exists(current_ckpt_dir):
                write_to_main_log(accelerator=self.accelerator, result=f"[Rank 0] Attempting to remove possibly incomplete checkpoint directory: {current_ckpt_dir}", type='warning')
                try:
                    shutil.rmtree(current_ckpt_dir)
                except Exception as clean_e:
                    write_to_main_log(accelerator=self.accelerator, result=f"[Rank 0] Error removing incomplete checkpoint directory: {clean_e}", type='error')

        finally:
            # Synchronize all ranks after the save attempt (successful or failed)
            self.accelerator.wait_for_everyone()

    def _extract_iteration_from_dirname(self, dirname):
        """
        Extract iteration number from directory name like 'iter_1250'
        Returns -1 if the format doesn't match expected pattern
        """
        match = re.match(r'iter_(\d+)', dirname)
        if match:
            return int(match.group(1))
        return -1

    def _clean_checkpoints(self):
        """
        Cleans old checkpoint directories, keeping only the latest `max_to_keep`.
        Only runs on the main process. Requires max_to_keep is configured.
        Uses directory names instead of metadata files.
        """
        if not self.is_main_process or self.max_to_keep is None:
            return

        try:
            base_dir = self.save_dir
            ckpt_dirs = []
            
            # Nothing to clean if dir doesn't exist
            if not os.path.isdir(base_dir):
                return 

            # List directories and extract iteration numbers from directory names
            for dname in os.listdir(base_dir):
                full_path = os.path.join(base_dir, dname)
                
                # Check if it's a directory and follows our naming pattern
                if os.path.isdir(full_path) and dname.startswith("iter_"):
                    # Extract iteration number from directory name
                    iter_num = self._extract_iteration_from_dirname(dname)
                    
                    if iter_num != -1:
                        # Check if directory contains checkpoint files to ensure it's valid
                        if any(fname.endswith(".safetensors") for fname in os.listdir(full_path)):
                            ckpt_dirs.append((iter_num, full_path))
                        else:
                            # Directory appears to be empty or incomplete
                            write_to_main_log(
                                accelerator=self.accelerator, 
                                result=f"[Rank 0] Found possibly incomplete checkpoint dir: {full_path}", 
                                type='warning'
                            )
                    
            if len(ckpt_dirs) <= self.max_to_keep:
                return

            # Sort by iteration number
            ckpt_dirs.sort(key=lambda x: x[0])

            # Identify checkpoints to delete (all except the last max_to_keep)
            checkpoints_to_delete = ckpt_dirs[:-self.max_to_keep]

            for iter_num, dir_path in checkpoints_to_delete:
                try:
                    shutil.rmtree(dir_path)
                    write_to_main_log(
                        accelerator=self.accelerator, 
                        result=f"[Rank 0] Deleted old checkpoint: {dir_path} (iteration {iter_num})"
                    )
                except OSError as e:
                    write_to_main_log(
                        accelerator=self.accelerator, 
                        result=f"[Rank 0] Error deleting {dir_path}: {e}", 
                        type='error'
                    )

        except Exception as e:
            write_to_main_log(accelerator=self.accelerator, result=f"[Rank 0] Error during backbone checkpoint cleaning process: {e}", type='error')
            import traceback
            write_to_main_log(accelerator=self.accelerator, result=traceback.format_exc(), type='error')

    
    def remove_prefix(self, full_teacher_state_dict):
        if not full_teacher_state_dict: 
            return None

        prefix = 'backbone.'
        gathered_backbone_state_dict = {
            k[len(prefix):]: v for k, v in full_teacher_state_dict.items() if k.startswith(prefix)
                }

        if not gathered_backbone_state_dict:
            del full_teacher_state_dict
            return None
        return gathered_backbone_state_dict