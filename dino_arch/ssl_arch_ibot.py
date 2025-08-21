from datetime import datetime
import math
import sys
from torch import nn
import torch

from torch.nn import functional as F
from data.collate import collate_data_and_cast
from utils.config import load_model_configs
from utils.dino_utils import ( 
    cancel_gradients_last_layer, 
    clip_gradients,  
    freeze_layers, 
    get_params_groups, 
    init_dino_training_models,
    init_schedulers 
) 
from losses.ibot_patch_loss import iBOTPatchLoss
from utils.logger_utils import write_to_main_log 


class SSLMetaArch(nn.Module):
    def __init__(
        self,
        config,
        accelerator, 
        dino_loss, 
        mask_generator=None
    ):
        super().__init__()
        self.config = config
        self.accelerator = accelerator 
        self.dino_loss = dino_loss 
        self.model_type = config.train.model_type
        self.model_params = load_model_configs(self.model_type) 
        self.device = accelerator.device
        self.mask_generator = mask_generator
         
        
        # Set up dimensions
        self.img_size = self.config.crops.global_crops_size
        self.patch_size = self.model_params.patch_size
        self.n_tokens = (self.img_size // self.patch_size) ** 2 
         
        # Initialize models
        self.student_model, self.teacher_model, self.model_config = init_dino_training_models(config=self.config , use_ibot=True, accelerator=self.accelerator   )
        
        # Set up iBOT (assuming always True) 
        self.teacher_layer_embeddings = None
        self.student_layer_embeddings = None  
        self.ibot_patch_loss = iBOTPatchLoss(self.config.ibot.out_dim)
        self.apply_interpolate= True if  self.model_params.model_type == 'vit' else False
        # Freeze layers if specified
        self.freeze_backbone_layers = getattr(config.train, 'freeze_backbone_layers', 0) 
        freeze_layers(num_layers_to_freeze=self.freeze_backbone_layers,   accelerator=self.accelerator,   student_model=self.student_model )
 
        # Initialize optimizer
        params_groups = get_params_groups(self.student_model)
        self.optimizer = torch.optim.AdamW(params_groups)
         
        # Set up schedulers
        self.lr_schedule,self.wd_schedule,self.momentum_schedule,self.teacher_temp = init_schedulers(config=config)
         
        # Prepare model and optimizer with accelerator
        self.student_model,self.teacher_model, self.optimizer = accelerator.prepare(
            self.student_model,self.teacher_model, self.optimizer, 
        )
          
    def _process_backbone_output(self, output): 
        if hasattr(output, "last_hidden_state"):
            cls_tokens = output.last_hidden_state[:, 0]
            patch_tokens = output.last_hidden_state[:, 1:]
        else:
            cls_tokens = output[:, 0]
            patch_tokens = output[:, 1:]
        return cls_tokens, patch_tokens
 

    def forward(self, iteration, imgs, crops): 
        self.optimizer.zero_grad()
 
        with self.accelerator.autocast():
            if self.mask_generator:
                crops = collate_data_and_cast(
                    cfg=self.config,
                    samples_list=crops,
                    n_tokens=self.n_tokens, 
                    mask_generator=self.mask_generator
                )
            # Update learning rate and weight decay according to schedules

            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.lr_schedule[iteration]
                if i == 0:
                    param_group["weight_decay"] = self.wd_schedule[iteration]
 
            teacher_temp = self.config.train.teacher_temp
            
            loss_dict = self._compute_loss(crops, teacher_temp)
            total_loss = loss_dict['total_loss']
            
            # Check for NaN loss
            if not math.isfinite(total_loss.item()):
                write_to_main_log( accelerator=self.accelerator, result="Loss is {}, stopping training".format(total_loss.item()), type='error')   
                sys.exit(1)
    
            self.accelerator.backward(total_loss)
                
                # Clip gradients if needed
            if self.config.train.clip_grad:
                clip_gradients(self.student_model, self.config.train.clip_grad)
                
                # Cancel gradients in last layer if needed
            cancel_gradients_last_layer(
                    iteration, 
                    self.student_model, 
                    self.config.train.freeze_last_layer
                )
                
            self.optimizer.step() 

        with torch.no_grad():
            self.m = self.momentum_schedule[iteration]
             
            for param_q, param_k in zip(
                self.student_model.parameters(),
                self.teacher_model.parameters()
            ):
                param_k.data.mul_(self.m).add_((1 - self.m) * param_q.data)

        # Ensure all processes are synced
        self.accelerator.wait_for_everyone()
 

        now = datetime.now()
         
        display_dict = {} 
        for key, value in loss_dict.items():
            if key == 'ibot_loss':
                display_dict[key] = value.item() / 2  
            else:
                display_dict[key] = value.item()



        display_total = 0
        if 'dino_local_loss' in display_dict:
            display_total += display_dict['dino_local_loss']
        if 'dino_global_loss' in display_dict:
            display_total += display_dict['dino_global_loss']
        if 'ibot_loss' in display_dict:
            display_total += display_dict['ibot_loss']  # Zaten 2'ye bölünmüş değer
        # Format loss names the same way as the original
        loss_strings = []
        if 'dino_local_loss' in display_dict:
            loss_strings.append(f"Local DINO: {display_dict['dino_local_loss']:.4f}")
        if 'dino_global_loss' in display_dict:
            loss_strings.append(f"Global DINO: {display_dict['dino_global_loss']:.4f}")
        if 'ibot_loss' in display_dict:
            loss_strings.append(f"iBOT: {display_dict['ibot_loss']:.4f}")
        
        # Get memory stats
        if torch.cuda.is_available():
            device = self.accelerator.device
            if device.type == 'cuda':
                total_memory = torch.cuda.get_device_properties(device.index).total_memory
                reserved_memory = torch.cuda.memory_reserved(device.index)
                total_memory_gb = total_memory / (1024 ** 3)
                reserved_memory_gb = reserved_memory / (1024 ** 3)
            else:
                total_memory_gb = reserved_memory_gb = 0
        else:
            total_memory_gb = reserved_memory_gb = 0
        
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
         
        result = 'Total Loss: {:.4f}\t{}\tLR: {:.6f}\tTeacher temp: {:.4f}\tTeacher momentum: {:.6f}\tIteration: {:5d}/{:5d}\tMemory: {:.2f}/{:.2f}'.format(
            total_loss,
            '\t'.join(loss_strings),
            self.optimizer.param_groups[0]["lr"],
            self.teacher_temp[iteration],
            self.m,
            int(iteration),
            int(self.config['train']['max_iterations']), 
            reserved_memory_gb,
            total_memory_gb 
        )
        
        return result

    @torch.no_grad()
    def _forward_teacher(self, crops, teacher_temp):
        
        global_crops = crops['collated_global_crops'].to(self.device)
        mask_indices_list = crops['mask_indices_list'].to(self.device)
        n_masked_patches_tensor = crops.get('n_masked_patches', None)
        if n_masked_patches_tensor is not None:
            n_masked_patches_tensor = n_masked_patches_tensor.to(self.device)
         
        n_global_crops = 2
        n_masked_patches = mask_indices_list.shape[0]
         
        teacher = self.teacher_model
        if hasattr(self.teacher_model, 'module'):
            teacher = self.accelerator.unwrap_model(self.teacher_model)
        if self.apply_interpolate: 
            teacher_output = teacher.backbone(global_crops,interpolate_pos_encoding=True)
        else: 
            teacher_output = teacher.backbone(global_crops )
        teacher_cls, teacher_patches = self._process_backbone_output(teacher_output)

        self.teacher_layer_embeddings = teacher_cls
        teacher_cls_chunks = teacher_cls.chunk(n_global_crops) 
        
        teacher_cls_shuffled = torch.cat((teacher_cls_chunks[1], teacher_cls_chunks[0]))
         
        teacher_dino_output = teacher.dino_head(teacher_cls_shuffled)
         
        teacher_masked_patches = torch.index_select(
            teacher_patches.flatten(0, 1),
            dim=0,
            index=mask_indices_list
        )
         
        if hasattr(teacher, 'ibot_head'): 
            teacher_ibot_output = teacher.ibot_head(teacher_masked_patches)
        else:
            teacher_ibot_output = teacher.dino_head(teacher_masked_patches)
         
        if hasattr(self.config.train, 'centering'):
            if self.config.train.centering == "centering": 
                teacher_dino_centered = self.dino_loss.softmax_center_teacher(
                    teacher_dino_output, teacher_temp=teacher_temp
                ).view(n_global_crops, -1, *teacher_dino_output.shape[1:])
                
                self.dino_loss.update_center(teacher_dino_output)
                 
                teacher_ibot_output = teacher_ibot_output.unsqueeze(0)
                teacher_ibot_centered = self.ibot_patch_loss.softmax_center_teacher(
                    teacher_ibot_output, teacher_temp=teacher_temp
                ).squeeze(0)
                self.ibot_patch_loss.update_center(teacher_ibot_output.squeeze(0))
            
            elif self.config.train.centering == "sinkhorn_knopp": 
                teacher_dino_centered = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_dino_output, teacher_temp=teacher_temp
                ).view(n_global_crops, -1, *teacher_dino_output.shape[1:])
                 
                teacher_ibot_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                    teacher_ibot_output, teacher_temp=teacher_temp,
                    n_masked_patches_tensor=n_masked_patches_tensor
                )
            else:
                teacher_dino_centered = teacher_dino_output.view(n_global_crops, -1, *teacher_dino_output.shape[1:])
                teacher_ibot_centered = teacher_ibot_output
        else:
            teacher_dino_centered = teacher_dino_output.view(n_global_crops, -1, *teacher_dino_output.shape[1:])
            teacher_ibot_centered = teacher_ibot_output
        
        return teacher_dino_centered, teacher_ibot_centered

    def _forward_student(self, crops): 
        global_crops = crops['collated_global_crops'].to(self.device)
        local_crops = crops['collated_local_crops'].to(self.device)
        mask_indices_list = crops['mask_indices_list'].to(self.device)
         
        unwrapped_student = self.accelerator.unwrap_model(self.student_model)

        if self.apply_interpolate:
            global_output = unwrapped_student.backbone(global_crops,interpolate_pos_encoding=True)
        else: 
            global_output = unwrapped_student.backbone(global_crops)

        global_cls, global_patches = self._process_backbone_output(global_output)
        self.student_layer_embeddings= global_cls

        if self.apply_interpolate:
            local_output = unwrapped_student.backbone(local_crops,interpolate_pos_encoding=True)
        else: 
            local_output = unwrapped_student.backbone(local_crops)

        local_cls, _ = self._process_backbone_output(local_output)
         
        global_dino = unwrapped_student.dino_head(global_cls)
        local_dino = unwrapped_student.dino_head(local_cls)
         
        student_masked_patches = torch.index_select(
            global_patches.flatten(0, 1),
            dim=0,
            index=mask_indices_list
        )
         
        if hasattr(unwrapped_student, 'ibot_head'):
            student_ibot_output = unwrapped_student.ibot_head(student_masked_patches)
        else:
            student_ibot_output = unwrapped_student.dino_head(student_masked_patches)
        
        return global_dino, local_dino, student_ibot_output

    def _compute_loss(self, crops, teacher_temp): 
        mask_indices_list = crops['mask_indices_list'].to(self.device)
        masks = crops['collated_masks'].to(self.device)
        masks_weight = crops['masks_weight'].to(self.device)
         
        loss_dict = {}
        n_global_crops = 2
        n_local_crops = self.config.crops.local_crops_number
        n_masked_patches = mask_indices_list.shape[0]
         
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops
        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
         
        ibot_loss_scale = 1.0 / n_global_crops
        loss_scales = 2   
         
        teacher_dino_centered, teacher_ibot_centered = self._forward_teacher(crops, teacher_temp)
         
        global_dino, local_dino, student_ibot_output = self._forward_student(crops)
          
 

        total_loss = 0.0
         
        if n_local_crops > 0: 
            local_dino_chunks = local_dino.chunk(n_local_crops)
             
            dino_local_loss = self.dino_loss(
                student_output_list=local_dino_chunks,
                teacher_out_softmaxed_centered_list=teacher_dino_centered
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            
            loss_dict['dino_local_loss'] = dino_local_loss
             
            dino_weight = getattr(self.config.dino_head, 'loss_weight', 1.0) 
            total_loss += dino_weight * dino_local_loss
         
        dino_global_loss = (
            self.dino_loss(
                student_output_list=[global_dino],
                teacher_out_softmaxed_centered_list=[teacher_dino_centered.flatten(0, 1)]
            ) 
            * loss_scales
            / (n_global_crops_loss_terms + n_local_crops_loss_terms)
        )
        
        loss_dict['dino_global_loss'] = dino_global_loss
         
        dino_weight = getattr(self.config.dino_head, 'loss_weight',1)
        total_loss += dino_weight * dino_global_loss
         
        if hasattr(self.config, 'ibot') and getattr(self.config.ibot, 'loss_weight', 0) > 0:
            ibot_loss = self.ibot_patch_loss.forward_masked(
                student_ibot_output,
                teacher_ibot_centered,
                student_masks_flat=masks,  
                n_masked_patches=n_masked_patches,
                masks_weight=masks_weight   
            )
             
            scaled_ibot_loss = ibot_loss * loss_scales * ibot_loss_scale
            
            loss_dict['ibot_loss'] = scaled_ibot_loss
             
            ibot_weight = getattr(self.config.ibot, 'loss_weight', 1.0)
            total_loss += ibot_weight * scaled_ibot_loss
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict
 