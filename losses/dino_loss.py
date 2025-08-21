import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
class DINOLoss(nn.Module):
    def __init__(self,config ):
        
        super().__init__()
        self.config=config
        out_dim=self.config.dino_head.out_dim
        self.n_global_crops = self.config.crops.global_crops_number
        self.ncrops = self.n_global_crops + self.config.crops.local_crops_number

        self.student_temp = self.config.train.student_temp
        self.center_momentum = self.config.train.center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        warmup_teacher_temp = self.config.train.warmup_teacher_temp
        teacher_temp= self.config.train.teacher_temp
        warmup_teacher_temp_iterations= self.config.train.warmup_teacher_temp_iterations
        max_iterations= self.config.train.max_iterations 
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_iterations),
            np.ones(max_iterations - warmup_teacher_temp_iterations) * teacher_temp
        ))
         
        
    def forward(self, student_output, teacher_output, iteration): 
        
        student_out = student_output / self.student_temp
        
        student_out = student_out.chunk(self.ncrops)
        
        

        temp = self.teacher_temp_schedule[iteration]
        
        
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        
        
        teacher_out = teacher_out.chunk(self.n_global_crops)
        
        
        global_loss = 0
        local_loss = 0
        n_global_terms = 0
        n_local_terms = 0
        
        
        total_loss = 0
        n_total_terms = 0
        
        for iq, q in enumerate(teacher_out):
            
            for v in range(len(student_out)):
                
                if v == iq:
                    continue
                
                
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                
                
                total_loss += loss.mean()
                n_total_terms += 1
                
                
                if v < self.n_global_crops:
                    global_loss += loss.mean()
                    n_global_terms += 1
                    
                else:
                    local_loss += loss.mean()
                    n_local_terms += 1
         
        total_loss /= n_total_terms
         
        if n_global_terms > 0:
            global_loss /= n_global_terms
        
        if n_local_terms > 0:
            local_loss /= n_local_terms
         
        self.update_center(teacher_output)
         
        return {
            'global_loss': global_loss,
            'local_loss': local_loss,
            'total_loss': total_loss
        }
        
    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (teacher_output.shape[0] * dist.get_world_size())
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)