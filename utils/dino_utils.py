# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
""" 
import copy
import importlib
import inspect 
import os 
import random 
import cv2
import numpy as np
import torch
from torch import   nn 
from PIL import ImageFilter, ImageOps, Image 
from transformers import AutoModel  
from accelerate.state import DistributedType  
from utils.distributed_checkpointer import DistributedCheckpointManager
from utils.config import load_model_configs  
from utils.logger_utils import write_to_main_log
from utils.model_checkpointer_ddp import DDPModelBackboneCheckpointManager 
from utils.post_processing import generate_pca_from_attention_map, get_attention_maps
from utils.vis_tools import save_feature_maps_plot, visualize_attention_pca, visualize_attention_pca_rgb 
from vit_models.vision_transformer import DINOiBOTWrapper, DinoWrapper, build_dino_head_from_config, build_ibot_head_from_config, build_transformer_model_from_config, build_transformer_model_from_timm
 

TIMM_LIST=['MahmoodLab/UNI2-h', 'MahmoodLab/UNI','prov-gigapath/prov-gigapath']
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img 
        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class ClaheFilter(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.2 ):
        self.prob = p 

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img  
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return Image.fromarray(clahe.apply(np.array(img)))



class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
 
 
def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6) 
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n: 
            p.grad = None

 


def import_all_functions_from_module(module_name): 
    module = importlib.import_module(module_name) 
    functions = {name: obj for name, obj in inspect.getmembers(module, inspect.isfunction)} 
    return functions

def call_function(functions, function_name, **kwargs):
    if function_name in functions: 
        return functions[function_name](**kwargs)
    else:
        raise ValueError(f"Architecture '{function_name}' not found.")
    
def cosine_scheduler(base_value, final_value, max_iters, warmup_iters=0, start_warmup_value=0, keep_constant_after_warmup=False):
    """
    Creates a scheduler with optional warmup and configurable post-warmup behavior.
    
    Args:
        base_value: Base value after warmup (peak value)
        final_value: Final value at the end of schedule
        max_iters: Total number of iterations
        warmup_iters: Number of warmup iterations
        start_warmup_value: Initial value for warmup
        keep_constant_after_warmup: If True, keeps value constant at base_value after warmup
    
    Returns:
        numpy array containing schedule values
    """
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    
    if keep_constant_after_warmup:
        # Keep constant at base_value after warmup
        remaining_iters = max_iters - warmup_iters
        constant_schedule = np.ones(remaining_iters) * base_value
        schedule = np.concatenate((warmup_schedule, constant_schedule))
    else:
        # Original cosine decay after warmup
        iters = np.arange(max_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        schedule = np.concatenate((warmup_schedule, schedule))
    
    assert len(schedule) == max_iters
    return schedule
 

 
def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

import torch

def freeze_layers(student_model,accelerator,   num_layers_to_freeze=0):
 
    if num_layers_to_freeze <= 0:
        return
     
     
    write_to_main_log(accelerator=accelerator, result=f"Attempting to freeze first {num_layers_to_freeze} layers/blocks in  model")
    
    backbone = student_model.backbone
      
    transformer = backbone.module if hasattr(backbone, "module") else backbone
        
        # Access encoder and layers
    if hasattr(transformer, "encoder"):
            encoder = transformer.encoder
            layers_attr = "layer" if hasattr(encoder, "layer") else "layers"
    elif hasattr(transformer, "blocks"):
            encoder = transformer
            layers_attr = "blocks"
    else: 
            write_to_main_log(accelerator=accelerator, result="Encoder structure not found!")
            return
    '''
    if hasattr(transformer, "embeddings"):
        for param in transformer.embeddings.parameters():
            param.requires_grad = False
    '''
        # Freeze layers
    layers = getattr(encoder, layers_attr)
    freeze_layers = min(num_layers_to_freeze, len(layers))
    for i in range(freeze_layers): 
            for param in layers[i].parameters():

                param.requires_grad = False
    
    # Calculate statistics if main process
    if accelerator.is_main_process:
        frozen_params = total_params = 0
        for param in student_model.parameters():
            total_params += param.numel()
            if not param.requires_grad:
                frozen_params += param.numel()
        
        frozen_percentage = (frozen_params / total_params) * 100
        write_to_main_log(accelerator=accelerator, result=f"Total: {total_params:,}, Frozen: {frozen_params:,} ({frozen_percentage:.2f}%)")


 
def init_dino_training_models(config, accelerator,   use_ibot=True ): 
 
    # Use regular dict instead of OrderedDict
    student_model_dict = {}
    teacher_model_dict = {}  
    model_params = load_model_configs(config.train.model_type) 
    

    level_awareness=getattr(config.train, "level_awareness", None)
    
    checkpoint_path = getattr(config.train, 'checkpoint_path', None)
    if checkpoint_path and config.train.use_pretrained: 
        if  os.path.exists(checkpoint_path):
            checkpoint_path=config.train.checkpoint_path
    
    student_backbone, student_config = build_transformer_model_from_config(
            load_pretrained=config.train.use_pretrained, 
            custom_config=model_params,
            model_type=config.train.model_type,
            accelerator=accelerator,
            checkpoint_path=checkpoint_path,
            level_awareness=level_awareness,
        )
        
    teacher_backbone, teacher_config = build_transformer_model_from_config(
            load_pretrained=config.train.use_pretrained, 
            custom_config=model_params,
            model_type=config.train.model_type,
            accelerator=accelerator,
            checkpoint_path=checkpoint_path,
            level_awareness=level_awareness
        ) 
    model_config= teacher_config 
    
    gradient_checkpointing_enable = getattr(config.train, "gradient_checkpointing_enable", False)
    if gradient_checkpointing_enable:
        if hasattr(student_backbone , 'gradient_checkpointing_enable'):
            student_backbone.gradient_checkpointing_enable() 
            write_to_main_log( accelerator=accelerator,result = "Gradient checkpointing enabled.")
    student_model_dict["backbone"] = student_backbone
    teacher_model_dict["backbone"] = teacher_backbone
    
    # Build DINO heads
    student_dino_head = build_dino_head_from_config(config=config, model_params=student_config)
    teacher_dino_head = build_dino_head_from_config(config=config, model_params=teacher_config)
    student_model_dict["dino_head"] = student_dino_head
    teacher_model_dict["dino_head"] = teacher_dino_head
    
    # Build iBOT heads if using separate heads
    if config.train.ibot_separate_head and use_ibot:
        student_ibot_head = build_ibot_head_from_config(config=config, model_params=student_config)
        teacher_ibot_head = build_ibot_head_from_config(config=config, model_params=teacher_config)
        student_model_dict["ibot_head"] = student_ibot_head
        teacher_model_dict["ibot_head"] = teacher_ibot_head
        
        write_to_main_log( accelerator=accelerator,result = "iBOT using separate head")
    elif not config.train.ibot_separate_head and use_ibot: 
        write_to_main_log( accelerator=accelerator,result = "iBOT sharing head with DINO")
    
    # Create ModuleDicts
    student_model = nn.ModuleDict(student_model_dict)
    teacher_model = nn.ModuleDict(teacher_model_dict)
    
    # Copy student weights to teacher
    for k in student_model.keys():
        teacher_model[k].load_state_dict(student_model[k].state_dict())
    
    # Freeze teacher parameters
    for p in teacher_model.parameters():
        p.requires_grad = False
    
    return student_model, teacher_model,model_config

 
  
def init_dino_evaluiaton_model(config, accelerator):
    """Initialize evaluation model with support for .safetensors"""
    model_type = config.train.model_type 
    model_path = config.train.vit_ckpt_path if hasattr(config.train, 'vit_ckpt_path') else None 
    use_pretrained= config.train.use_pretrained 
    base_model = None  
    adapter_path= os.path.join(model_path,'adapter_config.json') 
    if  model_type in TIMM_LIST:
        base_model, _ = build_transformer_model_from_timm(
            load_from_disk=not use_pretrained,
            accelerator=accelerator, 
            model_type=model_type,
            weights_path=model_path
        ) 
        write_to_main_log(accelerator=accelerator, result = f"Loaded Timm model: {model_type}")
        return base_model
    
    if not use_pretrained:  
            write_to_main_log(accelerator=accelerator, result = f"Building transformer. Load pretrained: {model_path}")
            return  AutoModel.from_pretrained(model_path)
    else:  

            checkpoint_path = os.path.join('checkpoints', model_type) 
            source_path = checkpoint_path if os.path.exists(checkpoint_path) else None
            print('source_path: ',source_path, model_type)
            if source_path: 
                return AutoModel.from_pretrained(source_path)
            
            write_to_main_log(accelerator=accelerator, result = f"Building transformer. Load pretrained: {model_type}")
            return AutoModel.from_pretrained(model_type) 
 
def print_model_info(accelerator, model):
    
    import torch
    import torch.distributed as dist
    
    # Get distributed info
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0
    
    # Get parameter counts for this GPU
    local_params = sum(p.numel() for p in model.parameters())
    local_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Gather parameter counts from all processes
    param_tensor = torch.tensor([local_params, local_trainable], dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu')
    gathered_tensors = [torch.zeros_like(param_tensor) for _ in range(world_size)]
    
    if world_size > 1:
        dist.all_gather(gathered_tensors, param_tensor)
    else:
        gathered_tensors[0] = param_tensor
    
    # Get memory usage for this GPU
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
    else:
        allocated_memory = total_memory = 0
    
    # Process results
    if rank == 0:
        per_gpu_params = [t[0].item() for t in gathered_tensors]
        per_gpu_trainable = [t[1].item() for t in gathered_tensors]
         
        is_sharded = False 
        for module in model.modules():
            if 'FullyShardedDataParallel' in str(type(module)):
                is_sharded = True
                break
        
        if is_sharded:
            total_params = sum(per_gpu_params)
            total_trainable = sum(per_gpu_trainable)
        else:
            # For replicated models (like DDP), just use the first GPU's count
            total_params = per_gpu_params[0]
            total_trainable = per_gpu_trainable[0]
         
        write_to_main_log(accelerator=accelerator, result="===== Model Parameter Distribution =====")
        write_to_main_log(accelerator=accelerator, result=f"Total parameters: {total_params:,}")
        write_to_main_log(accelerator=accelerator, result=f"Total trainable: {total_trainable:,} ({total_trainable/total_params*100:.1f}%)")
         
        write_to_main_log(accelerator=accelerator, result="Per-GPU Distribution:")
        for i, (params, trainable) in enumerate(zip(per_gpu_params, per_gpu_trainable)):
            write_to_main_log(accelerator=accelerator, result=f"GPU {i}: {params:,} params, {trainable:,} trainable ({trainable/params*100:.1f}%)")
     
    if torch.cuda.is_available():
        write_to_main_log(accelerator=accelerator, result=f"GPU {rank} memory: {allocated_memory:.2f}GB / {total_memory:.2f}GB ({allocated_memory/total_memory*100:.1f}%)")
    
    # Free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
  
def normalize_array( image_tensor): 
    orig_min = torch.amin(image_tensor)
    orig_max = torch.amax(image_tensor) 
    epsilon = 1e-7
    normalized_tensor = (image_tensor - orig_min) / (orig_max - orig_min + epsilon)
        
    return normalized_tensor

def get_periodic_train_checkpointer(config, model, optimizer, accelerator, max_to_keep=3): 
    
    train_checkpoint_dir = os.path.join(
        config.output_folders.main_output,
        config.train.model_name,
        config.output_folders.train_checkpoint_dir
    )
     
    checkpoint_manager = DistributedCheckpointManager(
        model=model,
        optimizer=optimizer,
        save_dir=train_checkpoint_dir,
        accelerator=accelerator,
        max_to_keep=max_to_keep,
    )
     
    start_iteration = checkpoint_manager.load_latest() + 1
    
    return start_iteration, checkpoint_manager


def get_periodic_backbone_checkpointer_ddp(config,model, accelerator,max_to_keep=3):
    train_checkpoint_dir = os.path.join(
                config.output_folders.main_output , 
                config.train.model_name,  
                  config.output_folders.teacher_checkpoint_dir
            )
     
    checkpoint_manager = DDPModelBackboneCheckpointManager( 
        model=model,
        config=config, 
        save_dir=train_checkpoint_dir,
        accelerator=accelerator,
        max_to_keep=max_to_keep, 
        )   
      
    return  checkpoint_manager
 
   
def generate_samples(accelerator,config,model_params, imgs, masks, iteration,backbone_model,num_register_tokens=0):
    imgs=normalize_array(imgs)

    level_awareness   =getattr(config.train, "level_awareness", None)
    if level_awareness: 
        if hasattr(backbone_model, "backbone"):  
            backbone_model=backbone_model.backbone
    if accelerator.is_main_process and config.train.generate_samples:
        try: 
            img_shape = (imgs.shape[-2], imgs.shape[-1])
            
            sample_out_path = os.path.join(
                config.output_folders.main_output , 
                config.train.model_name, 
                config.output_folders.training_samples_dir
            )
            img_idx = np.random.randint(0, imgs.shape[0])
            with torch.no_grad():  
                    
                att_maps,reg_tokens,img_shape = get_attention_maps(
                        device=accelerator.device,
                        model=backbone_model,
                        patch_size= model_params.patch_size,
                        norm_image=torch.unsqueeze(imgs[img_idx], dim=0),
                        num_register_tokens=num_register_tokens,
                        img_shape=img_shape
                    )  
                sample_out_path=os.path.join(sample_out_path,f'samples_{iteration}')
                save_feature_maps_plot(feature_maps=att_maps,output_dir=sample_out_path, filename_prefix='att_maps', figure_title='Attention Maps') 
                if reg_tokens is not None:
                    save_feature_maps_plot(feature_maps=reg_tokens,output_dir=sample_out_path) 
                    pca_multi_channel = generate_pca_from_attention_map( 
                            attentions=  reg_tokens,
                            img_shape=img_shape,
                            n_components= config.train.PCA_component,
                            patch_size= model_params.patch_size,
                            normalize_pca=False 
                        )  
                    if config.train.PCA_component==3: 
                        visualize_attention_pca_rgb(
                            denorm_img=imgs[img_idx].cpu(),
                            filtered_pca=pca_multi_channel,
                            patch_size=model_params.patch_size, 
                            save_dir=sample_out_path,
                            file_name=f"pca_rgb_register_{iteration}"  )
                    visualize_attention_pca(
                            denorm_img=imgs[img_idx].cpu(),  
                            mask=None if masks is None else masks[img_idx].cpu(),   
                            patch_size=model_params.patch_size,
                            filtered_pca=pca_multi_channel,
                            save_dir=sample_out_path,
                            file_name=f"pca_register_{iteration}"
                        )
                pca_multi_channel = generate_pca_from_attention_map(
                            #attentions= reg_tokens if reg_tokens is not None  else att_maps,
                            attentions=  att_maps,
                            img_shape=img_shape,
                            n_components= config.train.PCA_component,
                            patch_size= model_params.patch_size,
                            normalize_pca=False 
                        )  
                
                if config.train.PCA_component==3: 
                    visualize_attention_pca_rgb(
                            denorm_img=imgs[img_idx].cpu(),
                            filtered_pca=pca_multi_channel,
                            patch_size=model_params.patch_size, 
                            save_dir=sample_out_path,
                            file_name=f"pca_rgb_attention_{iteration}"  )
                visualize_attention_pca(
                            denorm_img=imgs[img_idx].cpu(),  # Move to CPU
                            mask=None if masks is None else masks[img_idx].cpu(),   
                            patch_size=model_params.patch_size,
                            filtered_pca=pca_multi_channel,
                            save_dir=sample_out_path,
                            file_name=f"pca_attention_{iteration}"
                        )
                    
                write_to_main_log(accelerator=accelerator, result= f"Generated sample visualization at iteration {iteration}")
                    
        except Exception as e: 
            write_to_main_log(accelerator=accelerator, result=f"Error generating sample at iteration {iteration}: {e}", type='error') 
    del backbone_model
    
    
def init_schedulers( config):
        # Initialize learning rate, weight decay, and momentum schedulers
    lr_schedule = cosine_scheduler(
            config['train']['lr'],
            config['train']['min_lr'],
            config['train']['max_iterations'],
            warmup_iters=config['train']['warmup_iterations'],
        )
        
    wd_schedule = cosine_scheduler(
            config['train']['weight_decay'],
            config['train']['weight_decay_end'],
            config['train']['max_iterations'],
        )
        
        # Momentum parameter increases to 1.0 during training with cosine schedule
    momentum_schedule = cosine_scheduler(
            config['train']['momentum_teacher'],
            1.0,
            config['train']['max_iterations']
        )
    teacher_temp = np.concatenate((
            np.linspace(config.train.warmup_teacher_temp, config.train.teacher_temp, config.train.warmup_teacher_temp_iterations ),
            np.ones(config.train.max_iterations - config.train.warmup_teacher_temp_iterations ) * config.train.teacher_temp
        ))
    return lr_schedule,wd_schedule,momentum_schedule,teacher_temp
  
