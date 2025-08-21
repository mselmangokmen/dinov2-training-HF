# DINOv2 Training Framework

A comprehensive PyTorch framework for training DINOv2 (Self-Distillation with No Labels) models using Hugging Face transformers. This framework supports both standard and custom model architectures with distributed training capabilities.

## Features

- 🚀 **Distributed Training**: Multi-GPU training support with PyTorch DDP
- 🔧 **Flexible Configuration**: YAML-based configuration system
- 🎯 **Multiple Model Support**: Pre-trained and custom DINOv2 architectures
- 📊 **Advanced Training**: Teacher-student paradigm with momentum updates
- 🖼️ **Multi-Scale Crops**: Global and local crop augmentations
- 💾 **Mixed Precision**: BF16 training for memory efficiency
- 📈 **Monitoring**: Built-in PCA visualization and sample generation

## Supported Models

### Standard DINOv2 Models
- `facebook/dinov2-small` - 6 attention heads, 384 hidden size
- `facebook/dinov2-base` - 12 attention heads, 768 hidden size  
- `facebook/dinov2-large` - 16 attention heads, 1024 hidden size
- `facebook/dinov2-giant` - 24 attention heads, 1536 hidden size

### DINOv2 with Registers
- `facebook/dinov2-with-registers-small`
- `facebook/dinov2-with-registers-base`
- `facebook/dinov2-with-registers-large`
- `facebook/dinov2-with-registers-giant`

### Original DINO Models
- `facebook/dino-vits16` - ViT Small with 16x16 patches
- `facebook/dino-vitb16` - ViT Base with 16x16 patches
- `facebook/dino-vits8` - ViT Small with 8x8 patches
- `facebook/dino-vitb8` - ViT Base with 8x8 patches

## Installation

```bash
# Clone the repository
git clone https://github.com/mselmangokmen/dinov2-training-HF.git
cd dinov2-training-HF

# Install dependencies
pip install torch torchvision transformers
pip install pyyaml tensorboard pillow numpy

# For distributed training
pip install accelerate
```

## Quick Start

### 1. Prepare Your Dataset

Before training, prepare your dataset in the following format (compatible with PyTorch's ImageFolder):
```
datasets/
├── mydataset/
│   ├── train/
│   │   ├── label0/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   │   ├── label1/
│   │   │   └── ...
│   │   └── ...
│   └── test/
│       ├── label0/
│       │   └── ...
│       └── ...
└── pathology_dataset/
    ├── train/
    └── test/
```

### 2. Configure Training

Edit the configuration file `configs/dino/rgb_training_config.yaml`:

```yaml
train:
  model_name: 'your_model_name'
  model_type: 'facebook/dinov2-base'  # Choose from supported models
  global_batch_size: 64
  max_iterations: 60000
  lr: 1e-4

dataset:
  dataset_path: 'datasets/mydataset'  # Path to dataset folder (contains train/ and test/)
  mean: [0.485, 0.456, 0.406]         # ImageNet means
  std: [0.229, 0.224, 0.225]          # ImageNet stds
```

### 3. Start Training

```bash 
# Multi-GPU and multi-node distributed training
torchrun --nnodes 1 --nproc-per-node 2 train_dino.py --train_config_file configs/dino/rgb_training_config.yaml
 
```

## Configuration Reference

### Model Configuration

```yaml
train:
  model_name: 'my_dinov2_model'           # Output model name
  model_type: 'facebook/dinov2-base'     # Base model architecture
  use_pretrained: True                    # Use pre-trained weights
  freeze_backbone_layers: 0               # Number of layers to freeze
  gradient_checkpointing_enable: false   # Enable gradient checkpointing
```

### Training Parameters

```yaml
train:
  global_batch_size: 64          # Total batch size across all GPUs
  max_iterations: 60000          # Maximum training iterations
  lr: 1e-4                       # Learning rate
  min_lr: 1e-5                   # Minimum learning rate (cosine schedule)
  weight_decay: 0.04             # Weight decay
  clip_grad: 3.0                 # Gradient clipping
  warmup_iterations: 10000       # Warmup iterations
```

### Teacher-Student Configuration

```yaml
train:
  teacher_temp: 0.04                    # Teacher temperature
  student_temp: 0.1                     # Student temperature
  warmup_teacher_temp: 0.04             # Teacher warmup temperature
  warmup_teacher_temp_iterations: 500   # Teacher warmup iterations
  momentum_teacher: 0.9992              # Teacher momentum update
  freeze_last_layer: 100                # Iterations to freeze last layer
```

### Dataset Configuration

```yaml
dataset:
  dataset_path: 'datasets/mydataset'  # Path to dataset folder (contains train/ and test/)
  shuffle: true                       # Shuffle dataset
  resize: false                       # Resize images
  mean: [0.485, 0.456, 0.406]        # Channel means for normalization
  std: [0.229, 0.224, 0.225]         # Channel stds for normalization
```

### Data Augmentation

```yaml
crops:
  global_crops_number: 2        # Number of global crops
  local_crops_number: 6         # Number of local crops
  global_crops_size: 224        # Global crop size
  local_crops_size: 98          # Local crop size
  global_crops_scale: [0.4, 1.0]  # Global crop scale range
  local_crops_scale: [0.05, 0.4]  # Local crop scale range
```

### Loss Configuration

```yaml
dino_head:
  out_dim: 65536              # Output dimension
  norm_last_layer: False      # Normalize last layer
  loss_weight: 1.0            # DINO loss weight

ibot:
  loss_weight: 1.0            # iBOT loss weight (0 to disable)
  out_dim: 65536              # iBOT output dimension
  mask_sample_probability: 0.5 # Probability of masking
  mask_ratio_min_max: [0.1, 0.5]  # Mask ratio range
```

## Output Structure

Training generates the following outputs:

```
outputs/
├── train_checkpoints/          # Student model checkpoints
├── teacher_checkpoints/        # Teacher model checkpoints
├── training_samples/           # Generated sample images
├── training_results/           # Final trained models
└── log_files/                 # Training logs and metrics
```

## Monitoring Training

### Tensorboard Logs
```bash
tensorboard --logdir outputs/log_files
```

### Generated Samples
The framework automatically generates PCA-visualized samples during training to monitor feature learning progress.

## Advanced Usage

### Custom Dataset Integration

The framework uses PyTorch's ImageFolder structure by default and expects `train/` and `test/` subdirectories within the dataset path:

```python
# Implement custom dataset loader
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        # Your dataset implementation
        # Note: Default implementation expects dataset_path/train/ and dataset_path/test/
        pass
```

### Fine-tuning Pre-trained Models

```yaml
train:
  use_pretrained: True
  freeze_backbone_layers: 8    # Freeze first 8 layers
  lr: 1e-5                     # Lower learning rate for fine-tuning
```

## Model Architecture Details

| Model | Hidden Size | Attention Heads | Layers | Parameters |
|-------|-------------|-----------------|--------|------------|
| DINOv2-small | 384 | 6 | 12 | ~22M |
| DINOv2-base | 768 | 12 | 12 | ~87M |
| DINOv2-large | 1024 | 16 | 24 | ~307M |
| DINOv2-giant | 1536 | 24 | 40 | ~1.1B |

## Performance Tips

1. **Batch Size**: Use the largest batch size that fits in GPU memory
2. **Learning Rate**: Scale learning rate with batch size (lr = base_lr * batch_size / 256)
3. **Gradient Checkpointing**: Enable for large models to save memory
4. **Mixed Precision**: Use BF16 for A100 GPUs, FP16 for others
5. **Data Loading**: Increase `num_workers` for faster data loading

## Troubleshooting

### Common Issues

**Out of Memory**
```yaml
# Reduce batch size or enable gradient checkpointing
train:
  global_batch_size: 32
  gradient_checkpointing_enable: true
```

**Slow Training**
```yaml
# Increase number of workers and enable mixed precision
train:
  num_workers: 16
distribution:
  mixed_precision: bf16
```

**NaN Loss**
```yaml
# Reduce learning rate and increase warmup
train:
  lr: 5e-5
  warmup_iterations: 20000
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [DINOv2 Paper](https://arxiv.org/abs/2304.07193) by Meta AI
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/) for the deep learning framework