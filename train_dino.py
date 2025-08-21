import argparse 

from accelerate.utils import (
    set_seed, 
    DistributedType, 
)
 
from dataloaders.datasetloader_rgb import RGBDatasetLoader
from dino_arch.main_dino_trainer import Trainer as MainTrainer
from utils.config import   setup as setup_config
from utils.dist_utils import  initialize_ddp_accelerator_from_config, print_cluster_info
from utils.logger_utils import setup_accelerate_logger, write_to_main_log
 
def main(config_file):
    set_seed(42)
    config = setup_config(config_file=config_file)  
    accelerator= initialize_ddp_accelerator_from_config(config=config)

    accelerator.wait_for_everyone()
    print_cluster_info(accelerator=accelerator)
    
    setup_accelerate_logger(accelerator, config)
     
    if accelerator.is_main_process: 
        write_to_main_log(accelerator=accelerator, result= "Starting DinoMX training")
        write_to_main_log(accelerator=accelerator, result= f"Running with {accelerator.num_processes} processes")
        write_to_main_log(accelerator=accelerator, result= f"Mixed precision: {accelerator.mixed_precision}")


    actual_dist_type = accelerator.state.distributed_type 

    device = accelerator.device
    world_size = accelerator.num_processes
    
    write_to_main_log(accelerator=accelerator, result= f"Accelerator initialized. Device: {device}, World Size: {world_size}")
    write_to_main_log(accelerator=accelerator, result= f"Enabled Distributed Strategy: {actual_dist_type}")
    
 
    train_data = RGBDatasetLoader(cfg=config).train_loader
        
    trainer = MainTrainer( config=config, accelerator=accelerator, train_data=train_data)
    trainer.train()
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Accelerate-powered RGB DINO training')
    parser.add_argument('--train_config_file', type=str, dest='train_config_file',
                        default='ssl_default_config', help='Configuration file name for training parameters')

    args = parser.parse_args()
    config_file = args.train_config_file


    main(config_file=config_file) 