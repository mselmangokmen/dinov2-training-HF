import torch 
from data.masking import generate_maskig_generator
from dino_arch.ssl_arch import SSLMetaArch as DDPdinov1SSL 
from dino_arch.ssl_arch_ibot import SSLMetaArch as DDPdinov2SSL 
from utils.config import load_model_configs   
from utils.dino_utils import generate_samples, get_periodic_backbone_checkpointer_ddp, get_periodic_train_checkpointer, print_model_info, write_to_main_log 
from losses.dino_loss import DINOLoss as DINOv1Loss
from losses.dinov2_loss import DINOV2Loss as DINOv2Loss   
from accelerate.state import DistributedType
 
from utils.logger_utils import write_to_node_logs  

class Trainer:
    def __init__(
        self,
        config,
        accelerator, 
        train_data , 
    ) -> None:
        self.config = config 
        self.accelerator = accelerator  # Add accelerator
        self.model_name = self.config.train.model_name 
        self.train_data = train_data  
        self.level_awareness   =getattr(self.config.train, "level_awareness", None)
        ibot = getattr(self.config, "ibot", None)
        ibot_loss_weight = getattr(ibot, "loss_weight", 0) if ibot is not None else 0
        self.do_ibot = ibot_loss_weight > 0

        self.model_params = load_model_configs(self.config.train.model_type) 
        print('model params: ', self.model_params)
        self.mask_generator=None
        self.dist_type= accelerator.state.distributed_type

        self.num_register_tokens = getattr(self.model_params, 'num_register_tokens', 0)
        
        if self.do_ibot:
                self.mask_generator= generate_maskig_generator(self.config, patch_size=self.model_params.patch_size)
                loss_fn = DINOv2Loss(config=config).cuda()   
        else:
                loss_fn = DINOv1Loss(config=config).cuda()
        #######################################################

        ###### DEFINE ARCH CLASS ACCORDING TO TRAINING TYPE #####
        if self.do_ibot: 
            if self.accelerator.is_main_process:
                write_to_main_log( accelerator=self.accelerator, result='DDP DINO Training with iBOT is activated.')
            self.ssl_model = DDPdinov2SSL(
                                        config=config,
                                        accelerator=accelerator, 
                                        dino_loss=loss_fn,
                                        mask_generator=self.mask_generator
                                    ).cuda() 

        else: 
            if self.accelerator.is_main_process:
                write_to_main_log( accelerator=self.accelerator, result='DDP DINO Training without iBOT is activated.')
            self.ssl_model = DDPdinov1SSL(
                                config=config,
                                accelerator=accelerator, 
                                dino_loss=loss_fn 
                            ).cuda()  
 
        print_model_info(accelerator=self.accelerator, model=  self.ssl_model.student_model)   
            
 
    def train(self): 
        max_iter = self.config.train.max_iterations 


        self.start_iteration , self.checkpoint_manager =  get_periodic_train_checkpointer( config=self.config, model=self.ssl_model, optimizer=self.ssl_model.optimizer ,accelerator=self.accelerator)
        if self.accelerator.state.distributed_type == DistributedType.MULTI_GPU:
            self.backbone_checkpoint_manager =  get_periodic_backbone_checkpointer_ddp( config=self.config, model=self.ssl_model,  accelerator=self.accelerator)
        
        if self.start_iteration > 0:
            write_to_main_log(accelerator=self.accelerator, result= f"Resuming training for model: {self.model_name} from iteration {self.start_iteration}") 
        else:
            write_to_main_log(accelerator=self.accelerator, result=f"Starting training for model: {self.model_name}")
            write_to_main_log(accelerator=self.accelerator, result=f"Model Configuration: {self.ssl_model.model_params}")
        
        train_data = self.accelerator.prepare(self.train_data) 
        iteration = self.start_iteration
        # Training loop
        for data in train_data: 
            masks = None
            if self.level_awareness: 
                imgs, crops, global_level,local_level = data[0],data[1],data[2] ,data[3] 
            else: 
                if len(data) >2:
                    imgs, crops, masks = data[0],data[1],data[2]
                else:
                    imgs, crops = data

            if iteration >= max_iter:
                break
            
            # Forward pass handled by ssl_model

            if self.level_awareness: 
                result = self.ssl_model(iteration, imgs, crops, global_level,local_level)
            else: 
                result = self.ssl_model(iteration, imgs, crops)
            # Log results 
            write_to_node_logs(accelerator=self.accelerator, result=result) 
             
            iteration += 1
             
            if self.config.train.saveckp_freq > 0 and (iteration % self.config.train.saveckp_freq) == 0 and iteration > 0: 
                self.checkpoint_manager.save(iteration=iteration) 
                self.backbone_checkpoint_manager.save(iteration=iteration) 
                if self.accelerator.state.distributed_type == DistributedType.MULTI_GPU and self.accelerator.is_main_process :  
                    teacher_model=self.ssl_model.teacher_model  
                    generate_samples(
                        accelerator=self.accelerator, 
                        config=self.config, 
                        model_params=self.model_params,  
                        num_register_tokens=self.num_register_tokens, 
                        imgs=imgs, masks=masks,iteration=iteration, backbone_model=self.accelerator.unwrap_model(teacher_model.backbone))
                    
                
                self.accelerator.wait_for_everyone()
        
 
        self.checkpoint_manager.save(iteration=iteration)     
        self.backbone_checkpoint_manager.save(iteration=iteration) 
        if self.accelerator.state.distributed_type == DistributedType.MULTI_GPU and self.accelerator.is_main_process  :  
            teacher_model=self.ssl_model.teacher_model
            generate_samples(
                        accelerator=self.accelerator, 
                        config=self.config, 
                        num_register_tokens=self.num_register_tokens, 
                        model_params=self.model_params,  
                        imgs=imgs, masks=masks,iteration=iteration, backbone_model=self.accelerator.unwrap_model(teacher_model.backbone))
            write_to_main_log(accelerator=self.accelerator, result=f"Training completed at iteration {iteration}") 

        
        self.accelerator.wait_for_everyone()