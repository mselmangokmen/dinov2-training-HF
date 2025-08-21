from torchvision import transforms
from utils.dino_utils import GaussianBlur, Solarization
from torchvision.transforms.functional import InterpolationMode

class DataAugmentationDINO(object):
    def __init__(self, cfg):
        local_crops_number = cfg.crops.local_crops_number
        local_crops_scale = cfg.crops.local_crops_scale
        global_crops_scale = cfg.crops.global_crops_scale
        local_crops_size = cfg.crops.local_crops_size
        global_crops_size = cfg.crops.global_crops_size
        self.local_crops_number = local_crops_number
        mean=cfg.dataset.mean
        std=cfg.dataset.std
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        
        # transformation for the local small crops
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_crops_size, scale=local_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])
    
    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        
        # Format output dictionary to match your required structure
        output = {
            "global_crops": [crops[0], crops[1]],
            "local_crops": crops[2:], 
        }
        
        return output