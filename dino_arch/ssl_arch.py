from datetime import datetime
import math
import sys 
from torch import nn 
import torch 
 
from utils.config import load_model_configs
from utils.dino_utils import ( 
    cancel_gradients_last_layer, 
    clip_gradients,  
    get_params_groups,  
     freeze_layers, 
    init_dino_training_models,
    init_schedulers
)
from utils.logger_utils import write_to_main_log 
class SSLMetaArch(nn.Module):
    def __init__(
        self,
        config,
        accelerator, 
        dino_loss, 
    ) -> None:
        super().__init__()
        self.config = config
        self.accelerator = accelerator 
        self.dino_loss = dino_loss 
        self.model_type = config.train.model_type
        self.model_params = load_model_configs(self.model_type) 
        self.device = accelerator.device 
         
        
        self.student_model, self.teacher_model, self.model_config = init_dino_training_models(config=self.config , use_ibot=False, accelerator=self.accelerator )
         
        self.apply_interpolate= True if  self.model_params.model_type == 'vit' else False  
        freeze_backbone_layers = getattr(config.train, 'freeze_backbone_layers', 0)
        if freeze_backbone_layers > 0:
            freeze_layers(student_model=self.student_model,  accelerator=self.accelerator,    num_layers_to_freeze=freeze_backbone_layers)
 
        params_groups = get_params_groups(self.student_model)
        self.optimizer = torch.optim.AdamW(params_groups)
         
        self.lr_schedule,self.wd_schedule,self.momentum_schedule,self.teacher_temp = init_schedulers(config=config)
         
        self.student_model,self.teacher_model, self.optimizer = accelerator.prepare(
            self.student_model,self.teacher_model, self.optimizer
        )
        
         
   
    def _forward_model(self, crops, model,interpolate_pos_encoding=False): 
        if hasattr(model, "module"):
            model = model.module
        if interpolate_pos_encoding:
                output = model.backbone(crops, interpolate_pos_encoding=interpolate_pos_encoding)
        else: 
                output = model.backbone(crops)

        if hasattr(output, "last_hidden_state"):
            backbone_out = output.last_hidden_state[:, 0]
        else:
            backbone_out = output
            
        return model.dino_head(backbone_out)  


    @torch.no_grad()
    def _forward_teacher_output(self, crops):
        collated_global_crops = torch.cat(crops["global_crops"], dim=0) 
 
        return self._forward_model(collated_global_crops, self.teacher_model)

    def _forward_student_output(self, crops):
        collated_global_crops = torch.cat(crops["global_crops"], dim=0) 
        student_global_out = self._forward_model(collated_global_crops, self.student_model,interpolate_pos_encoding=self.apply_interpolate)

        collated_local_crops = torch.cat(crops["local_crops"], dim=0) 

        student_local_out = self._forward_model(collated_local_crops, self.student_model,interpolate_pos_encoding= self.apply_interpolate)
        student_out = torch.cat([student_global_out, student_local_out], dim=0)
        return student_out

    def _forward(self, crops):
        student_out = self._forward_student_output(crops)
        teacher_out = self._forward_teacher_output(crops)
        return student_out, teacher_out
    
    def forward(self, iteration, imgs, crops): 
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[iteration]
            if i == 0:
                param_group["weight_decay"] = self.wd_schedule[iteration]

        self.optimizer.zero_grad()
        with self.accelerator.autocast():
            student_output, teacher_output = self._forward(crops)
            loss = self.dino_loss(student_output, teacher_output, iteration)

            if not math.isfinite(loss['total_loss'].item()):
                write_to_main_log( accelerator=self.accelerator, result="Loss is {}, stopping training".format(total_loss.item()), type='error')  
                sys.exit(1)
    
            self.param_norms = None
            
            self.accelerator.backward(loss['total_loss'])
                
            if self.config['train']['clip_grad']:
                self.param_norms = clip_gradients(self.student_model, self.config['train']['clip_grad'])
                
            cancel_gradients_last_layer(
                    iteration, 
                    self.student_model, 
                    self.config['train']['freeze_last_layer']
                )
                
            self.optimizer.step() 

        # Update teacher model with EMA
        with torch.no_grad():
            self.m = self.momentum_schedule[iteration]
            
            # Copy parameters from student model (with unwrapping for DDP)
            for param_q, param_k in zip(
                self.student_model.parameters(),
                self.teacher_model.parameters()
            ):
                param_k.data.mul_(self.m).add_((1 - self.m) * param_q.data)

        # Ensure all processes are synced
        self.accelerator.wait_for_everyone()

        # Format and return training status
        now = datetime.now()
        total_loss = loss['total_loss'].item()

        loss_strings = []
        for key, value in loss.items():
            if key != 'total_loss':
                loss_strings.append(f"{key.replace('_', ' ').title()}: {value.item():.4f}")

        # Get memory statistics
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