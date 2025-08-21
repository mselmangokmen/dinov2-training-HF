import copy
import os
import argparse
import logging
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import transforms  
 
from accelerate.utils import set_seed 
from sklearn.metrics import  precision_recall_fscore_support,confusion_matrix,silhouette_score,adjusted_mutual_info_score
 
  
from dataloaders.datasetloader_rgb import RGBDatasetLoader  
from utils.dino_utils import  init_dino_evaluiaton_model
from utils.dist_utils import initialize_ddp_accelerator_from_config  
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib 

# Force matplotlib to use non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns 


def load_config(config_name: str):
    """Load configuration file."""
    config_path = os.path.join('configs', f"{config_name}.yaml")
    omgConf= OmegaConf.load(config_path)
    return OmegaConf.create(omgConf) 

def get_device():
    """Get the most suitable device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


class ComprehensiveEvaluator:


    def visualize_embeddings_kmeans(self,features, labels ):
        """Visualize embeddings using K-means clustering in 2D space"""
        if self.accelerator.is_main_process:
            self.logger.info("Visualizing embeddings with K-means clustering...")

        # Extract features
        #features, labels = self.extract_features(self.val_loader)

        if self.accelerator.is_main_process:
            try:
                # Get class names from the dataset
                class_names = None
                if hasattr(self.val_loader.dataset, 'get_classes'):
                    class_names = self.val_loader.dataset.get_classes()
                elif hasattr(self.val_loader.dataset, 'classes'):
                    class_names = self.val_loader.dataset.classes
                elif hasattr(self.val_loader.dataset, 'base_dataset') and hasattr(self.val_loader.dataset.base_dataset, 'classes'):
                    class_names = self.val_loader.dataset.base_dataset.classes
                
                # Create a mapping from numeric labels to class names
                idx_to_class = {}
                if class_names:
                    for i, name in enumerate(class_names):
                        idx_to_class[i] = name
                else:
                    # If we can't find class names, use numeric labels
                    unique_labels = np.unique(labels)
                    for label in unique_labels:
                        idx_to_class[label] = f"Class {label}"
                
                # Perform dimensionality reduction to 2D for visualization
                from sklearn.decomposition import PCA
                from sklearn.manifold import TSNE
                
                # First apply PCA to reduce dimensions
                n_components_pca = min(30, features.shape[1])
                if features.shape[1] > n_components_pca:
                    self.logger.info(f"Reducing dimensions from {features.shape[1]} to {n_components_pca} with PCA before t-SNE...")
                    pca = PCA(n_components=n_components_pca)
                    features_reduced = pca.fit_transform(features)
                else:
                    features_reduced = features
                
                # Apply t-SNE for better visualization
                self.logger.info("Applying t-SNE for 2D visualization...")
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
                features_2d = tsne.fit_transform(features_reduced)
                
                # Perform k-means clustering
                self.logger.info(f"Performing K-means clustering with {self.num_classes} clusters...")
                kmeans = KMeans(n_clusters=self.num_classes, random_state=42)
                cluster_labels = kmeans.fit_predict(features)
                
                # Get cluster centers and project them to 2D space (for right plot only)
                centers_2d = np.zeros((self.num_classes, 2))
                for i in range(self.num_classes):
                    cluster_points = features_2d[cluster_labels == i]
                    if len(cluster_points) > 0:  # Ensure the cluster is not empty
                        centers_2d[i] = np.mean(cluster_points, axis=0)
                
                # Create a figure with two subplot columns
                fig = plt.figure(figsize=(20, 10))
                
                # Create subplot layout with more space for the main plots
                gs = plt.GridSpec(2, 2, height_ratios=[4, 1])
                
                # Plot 1: Points colored by true class labels (WITHOUT cluster centers)
                ax1 = fig.add_subplot(gs[0, 0])
                scatter1 = ax1.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, 
                                    cmap='tab10', alpha=0.6, marker='.')
                ax1.set_title('2D Embeddings with True Class Labels', fontsize=14)
                
                # Create legend subplot for true class names
                ax_legend1 = fig.add_subplot(gs[1, 0])
                ax_legend1.axis('off')
                
                # Get unique labels and sort them
                unique_true_labels = np.sort(np.unique(labels))
                
                # Create a legend with actual class names
                legend_entries = []
                for i, label_idx in enumerate(unique_true_labels):
                    color = scatter1.cmap(scatter1.norm(label_idx))
                    class_name = idx_to_class.get(label_idx, f"Class {label_idx}")
                    legend_entry = ax_legend1.text(0.1 + (i % 5) * 0.18, 0.6 - (i // 5) * 0.3,
                                                f"● {class_name}", 
                                                color=color, fontsize=12, ha='left', va='center')
                    legend_entries.append(legend_entry)
                
                # Add title to the legend
                ax_legend1.set_title("Class Name Mapping", fontsize=14)
                
                # Plot 2: Points colored by predicted clusters (WITH cluster centers but NO legend)
                ax2 = fig.add_subplot(gs[0, 1])
                scatter2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels, 
                                    cmap='tab10', alpha=0.6, marker='.')
                ax2.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', 
                        marker='X', s=200, label='Cluster Centers')
                ax2.set_title('K-means Clustering Results', fontsize=14)
                
                # Create empty subplot for right bottom (no legend)
                ax_legend2 = fig.add_subplot(gs[1, 1])
                ax_legend2.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'kmeans_embeddings_visualization.png'), dpi=300)
                plt.close()
                
                # Calculate metrics about the clustering
                ami = adjusted_mutual_info_score(labels, cluster_labels)
                
                # Check if all points are in one cluster to avoid division by zero in silhouette score
                unique_clusters = len(np.unique(cluster_labels))
                if unique_clusters > 1 and unique_clusters < len(features):
                    silhouette = silhouette_score(features, cluster_labels)
                else:
                    silhouette = 0
                    self.logger.warning("Silhouette score could not be computed (all samples in same cluster or each sample in its own cluster)")
                
                # Return metrics for potential inclusion in JSON output
                kmeans_results = {
                    'adjusted_mutual_info': ami,
                    'silhouette_score': silhouette,
                    'num_clusters': self.num_classes,
                    'class_names': [idx_to_class.get(i, f"Class {i}") for i in range(self.num_classes)]
                }
                
                return kmeans_results
                
            except Exception as e:
                self.logger.error(f"Failed to visualize embeddings with K-means: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                return {
                    'error': str(e),
                    'adjusted_mutual_info': 0,
                    'silhouette_score': 0
                }
        
        return None
    
    
    def __init__(self, config, output_dir='evaluation_outputs'):
        """
        Initialize the evaluator using the provided configuration.
        
        Args:
            config: The loaded configuration object
            output_dir: Directory to save evaluation results
        """
        self.config = config
        self.model_path = config.train.vit_ckpt_path if hasattr(config.train, 'vit_ckpt_path') else None
        self.model_type = config.train.model_type  
        
        # Initialize accelerator with mixed precision from config
        mixed_precision = 'bf16' 
        
        self.accelerator = initialize_ddp_accelerator_from_config(config=config) 
        # Configure logging
        self.logger = logging.getLogger(__name__)
        if self.accelerator.is_main_process:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
        else:
            self.logger.disabled = True

        self.output_dir = output_dir
        if self.accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True) 

        # Call the init_models function to load and initialize the model
        if self.accelerator.is_main_process:
            self.logger.info("Starting model initialization using init_models...")
        
        self.model = init_dino_evaluiaton_model(config=self.config, accelerator=self.accelerator)
        self.level_awareness   =getattr(config.train, "level_awareness", None) 
        self.num_register_tokens= getattr(self.config.train,'num_register_tokens',4)
        self.use_registers= getattr(self.config.train,'use_registers',False)
        #print('num_register_tokens: ', self.num_register_tokens)
        #print('use_registers: ', self.use_registers)
        self.model.eval()
        # Get image size from config or use default
        #self.image_size = 224
        
        #if self.accelerator.is_main_process:
        #    self.logger.info(f"Using image size: {self.image_size}")

        # Setup transforms using parameters from config
 
 

        # Setup data using batch size from config
        self.batch_size = config.train.batch_size_per_gpu 
        self.train_loader, self.val_loader = self.setup_data(cfg=self.config)

        # Set number of classes for the dataset
        self.num_classes = config.train.num_classes

        # Prepare model
        self.model = self.accelerator.prepare(self.model)
    def setup_data(self, cfg):  
        data_loader = RGBDatasetLoader(cfg,   is_dino=False, is_infsampler=False) 

        train_loader, val_loader = data_loader.get_loaders() 

        return self.accelerator.prepare(train_loader, val_loader)



    def extract_features(self, loader):
        """Extract features from the model"""
        features = []
        labels = []
        self.model.eval()

        # Progress bar only on main process
        progress_bar = tqdm(
            loader,
            desc="Extracting features",
            disable=not self.accelerator.is_main_process,
            total=len(loader)   
        )

        with torch.no_grad():
            for i, batch_data in enumerate(progress_bar):
                #(images, batch_labels)
                if self.level_awareness:
                    images, batch_labels,global_level=batch_data[0],batch_data[1],batch_data[2]
                else:
                    images, batch_labels=batch_data[0],batch_data[1]
                
                if self.accelerator.is_main_process:
                    progress_bar.set_postfix({'batch': f'{i+1}/{len(loader)}'}) 
                # Forward pass
                try:
                    if self.level_awareness:
                        outputs = self.model(images,global_level)
                    else:
                        outputs = self.model(images)
                except Exception as e:
                    self.logger.error(f"Error in model forward pass: {str(e)}")
                    raise
                
                if hasattr(outputs, 'last_hidden_state'):
                    if self.use_registers:
                        #start_index_reg = 1     
                        #end_index_reg = 1 + self.num_register_tokens
                        register_token_embeddings = outputs.last_hidden_state[:, 1:1 + self.num_register_tokens, :]
                        #batch_features = register_token_embeddings.mean(dim=1)
                        #print('register_token_embeddings shape: ', register_token_embeddings.shape)
                        batch_features = register_token_embeddings[0]
                        #print('using registers mean')
                    else:
                        batch_features = outputs.last_hidden_state[:, 0]  # CLS token
                    #print('in last_hidden_state')
                elif hasattr(outputs, 'logits') and not isinstance(outputs, tuple):
                    batch_features = outputs.logits
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        batch_features = outputs.hidden_states[-1][:, 0] if batch_features.shape[1] == self.num_classes else outputs.hidden_states[-1].mean(dim=1)
                    #print('in logits')
                else:
                    # Handle standard tensor or tuple outputs
                    if isinstance(outputs, tuple):
                        batch_features = outputs[0]
                    elif isinstance(outputs, torch.Tensor):
                        batch_features = outputs
                    else:
                        # Try to find feature tensor
                        for attr_name in dir(outputs):
                            attr = getattr(outputs, attr_name)
                            if isinstance(attr, torch.Tensor) and attr.dim() >= 2:
                                batch_features = attr
                                break
                        else:
                            raise ValueError("Could not extract features from output")

                # Reshape if needed
                if batch_features.dim() > 2:
                    batch_features = batch_features.mean(dim=list(range(1, batch_features.dim() - 1)))
                    if batch_features.dim() > 2:
                        batch_features = batch_features.view(batch_features.size(0), -1)
                features.append(batch_features.cpu())
                try: 
                    labels.append(batch_labels.cpu())
                except Exception as e:  
                    labels.append(batch_labels)
        # Concatenate results
        features = torch.cat(features)
        labels = torch.cat(labels)

        return features.numpy(), labels.numpy()
    def linear_probe(self,train_features,train_labels,val_features,val_labels):
        """Evaluate features using linear probing"""
        if self.accelerator.is_main_process:
            self.logger.info("Starting linear probe evaluation...")

        # Extract features
        #train_features, train_labels = self.extract_features(self.train_loader)
        #val_features, val_labels = self.extract_features(self.val_loader)

        # Convert to tensors and move to accelerator device
        train_features = torch.FloatTensor(train_features).to(self.accelerator.device)
        train_labels = torch.LongTensor(train_labels).to(self.accelerator.device)
        val_features = torch.FloatTensor(val_features).to(self.accelerator.device)
        val_labels = torch.LongTensor(val_labels).to(self.accelerator.device)

        # Create classifier
        '''
        classifier = nn.Sequential(
            nn.Linear(train_features.shape[1], 2*(train_features.shape[1]//3)),
            nn.ReLU(),   
            nn.Linear(2*(train_features.shape[1]//3), train_features.shape[1]//2),
            nn.ReLU(),
            nn.Linear(train_features.shape[1]//2, train_features.shape[1]//3),
            nn.ReLU(),
            nn.Linear(train_features.shape[1]//3, self.num_classes)   
        ).to(self.accelerator.device)
        classifier = nn.Sequential(
            nn.Linear(train_features.shape[1], 2*(train_features.shape[1]//3)),
            nn.ReLU(),   
            nn.Linear(2*(train_features.shape[1]//3), train_features.shape[1]//2),
            nn.ReLU(),
            nn.Linear(train_features.shape[1]//2, self.num_classes)
        ).to(self.accelerator.device)
        '''
        classifier = nn.Linear(train_features.shape[1], self.num_classes).to(self.accelerator.device)
        optimizer = torch.optim.Adam(classifier.parameters())

        # Prepare with accelerator
        classifier, optimizer, train_features, train_labels, val_features, val_labels = \
            self.accelerator.prepare(
                classifier, optimizer, train_features, train_labels, val_features, val_labels
            )

        best_acc = 0
        patience = 5
        patience_counter = 0

        # Training loop
        for epoch in range(self.config.train.epochs):
            classifier.train()
            optimizer.zero_grad() 
            # Forward pass with accelerator-prepared tensors
            logits = classifier(train_features)
            loss = F.cross_entropy(logits, train_labels)
 
            self.accelerator.backward(loss)
            optimizer.step()

            # Validation
            classifier.eval()
            with torch.no_grad():
                val_logits = classifier(val_features)
                val_preds = torch.argmax(val_logits, dim=1)

                # Gather predictions and labels
                gathered_preds = self.accelerator.gather(val_preds)
                gathered_labels = self.accelerator.gather(val_labels)

                # Compute accuracy on main process
                if self.accelerator.is_main_process:
                    accuracy = (gathered_preds == gathered_labels).float().mean().item()
                else:
                    accuracy = 0.0

                # Broadcast accuracy to all processes
                accuracy_tensor = torch.tensor(accuracy, device=self.accelerator.device)
                dist.broadcast(accuracy_tensor, 0)
                accuracy = accuracy_tensor.item()

                if accuracy > best_acc:
                    best_acc = accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

        # Final evaluation
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_features)
            val_preds = torch.argmax(val_logits, dim=1)

            # Gather final predictions and labels
            gathered_preds = self.accelerator.gather(val_preds)
            gathered_labels = self.accelerator.gather(val_labels)

            # Convert to numpy for metric computation
            final_preds = gathered_preds.cpu().numpy()
            final_labels = gathered_labels.cpu().numpy()

            # Compute metrics on main process
            if self.accelerator.is_main_process:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    final_labels, final_preds, average='weighted'
                )

                # Generate and save confusion matrix
                if self.accelerator.is_main_process:
                    try:
                        cm = confusion_matrix(final_labels, final_preds)
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                        plt.xlabel('Predicted')
                        plt.ylabel('True')
                        plt.title('Confusion Matrix - Linear Probe')
                        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix_linear.png'))
                        plt.close()
                    except Exception as e:
                        self.logger.warning(f"Failed to save confusion matrix: {str(e)}")
            else:
                precision = recall = f1 = 0.0

            # Broadcast metrics to all processes
            metrics = torch.tensor([precision, recall, f1], device=self.accelerator.device)
            dist.broadcast(metrics, 0)
            precision, recall, f1 = metrics.tolist()

        return {
            'accuracy': best_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

 
    def knn_eval(self,train_features,train_labels,val_features,val_labels, k=5):
        """Evaluate features using k-NN"""
        if self.accelerator.is_main_process:
            self.logger.info("Starting k-NN evaluation...")

        # Extract features
        #train_features, train_labels = self.extract_features(self.train_loader)
        #val_features, val_labels = self.extract_features(self.val_loader)

        # Convert to tensors and move to accelerator device
        train_features = torch.FloatTensor(train_features).to(self.accelerator.device)
        val_features = torch.FloatTensor(val_features).to(self.accelerator.device)

        # Prepare tensors with accelerator
        train_features, val_features = self.accelerator.prepare(train_features, val_features)

        # Compute pairwise distances
        dist_matrix = torch.cdist(val_features, train_features)

        # Get top-k nearest neighbors
        _, indices = dist_matrix.topk(k, dim=1, largest=False)

        # Move indices to CPU for numpy operations
        indices_cpu = indices.cpu().numpy()

        if self.accelerator.is_main_process:
            # Get neighbor labels and make predictions
            neighbor_labels = train_labels[indices_cpu]
            predictions = np.array([
                np.bincount(neighbor_labels[i]).argmax()
                for i in range(len(val_labels))
            ])

            # Compute metrics
            accuracy = (predictions == val_labels).mean()
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_labels, predictions, average='weighted'
            )

            # Generate and save confusion matrix for KNN
            try:
                cm = confusion_matrix(val_labels, predictions)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix - {k}-NN')
                plt.savefig(os.path.join(self.output_dir, f'confusion_matrix_knn_{k}.png'))
                plt.close()
            except Exception as e:
                self.logger.warning(f"Failed to save KNN confusion matrix: {str(e)}")
        else:
            accuracy = precision = recall = f1 = 0.0

        # Broadcast metrics to all processes
        metrics = torch.tensor(
            [accuracy, precision, recall, f1],
            device=self.accelerator.device
        )
        dist.broadcast(metrics, 0)
        accuracy, precision, recall, f1 = metrics.tolist()

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def clustering_quality(self,features,labels):
        """Evaluate clustering quality of features"""
        if self.accelerator.is_main_process:
            self.logger.info("Evaluating clustering quality...")

        # Extract features
        #features, labels = self.extract_features(self.val_loader)

        if self.accelerator.is_main_process:
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=self.num_classes, random_state=42)
            cluster_labels = kmeans.fit_predict(features)

            # Compute metrics
            silhouette = silhouette_score(features, cluster_labels)
            ami = adjusted_mutual_info_score(labels, cluster_labels)

            # Save cluster distribution
            try:
                plt.figure(figsize=(12, 6))

                # Plot true class distribution
                plt.subplot(1, 2, 1)
                # Convert labels to integers to avoid the categorical warning
                label_counts = np.bincount(labels.astype(int))
                plt.bar(np.arange(len(label_counts)), label_counts)
                plt.title('True Class Distribution')
                plt.xlabel('Class')
                plt.ylabel('Count')

                # Plot cluster distribution
                plt.subplot(1, 2, 2)
                cluster_counts = np.bincount(cluster_labels.astype(int))
                plt.bar(np.arange(len(cluster_counts)), cluster_counts)
                plt.title('Cluster Distribution')
                plt.xlabel('Cluster')
                plt.ylabel('Count')

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'cluster_distribution.png'))
                plt.close()
            except Exception as e:
                self.logger.warning(f"Failed to save cluster distribution: {str(e)}")
        else:
            silhouette = ami = 0.0

        # Broadcast metrics to all processes
        if dist.is_initialized():
            metrics = torch.tensor([silhouette, ami], device=self.accelerator.device)
            dist.broadcast(metrics, 0)
            silhouette, ami = metrics.tolist()

        return {
            'silhouette': silhouette,
            'ami': ami
        }



    def feature_robustness(self, num_samples=1000):
        """Evaluate feature robustness to augmentations"""
        if self.accelerator.is_main_process:
            self.logger.info("Evaluating feature robustness...")

        # Create augmentation pipeline
        augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        ])

        # Sample subset of validation data
        dataset = self.val_loader.dataset
        indices = torch.randperm(len(dataset))[:num_samples]
        indices = [int(idx.item()) for idx in indices]
        similarities = []
        self.model.eval()

        with torch.no_grad():
            for idx in tqdm(indices, disable=not self.accelerator.is_main_process): 
                img, _ = dataset[idx]
 
                orig_img = img.unsqueeze(0)
                aug_img = augment(img).unsqueeze(0)

                # Move to correct device and prepare with accelerator
                orig_img, aug_img = self.accelerator.prepare(orig_img, aug_img)

                # Get features for original image
 
                orig_outputs = self.model(orig_img)
                # Enhanced handling of different output types
                if hasattr(orig_outputs, 'pooler_output'):
                    orig_feat = orig_outputs.pooler_output
                elif hasattr(orig_outputs, 'last_hidden_state'):
                    orig_feat = orig_outputs.last_hidden_state[:, 0]
                elif hasattr(orig_outputs, 'logits') and not isinstance(orig_outputs, tuple):
                    orig_feat = orig_outputs.logits
                    if hasattr(orig_outputs, 'hidden_states') and orig_outputs.hidden_states is not None:
                        orig_feat = orig_outputs.hidden_states[-1][:, 0] if orig_feat.shape[1] == self.num_classes else \
                        orig_outputs.hidden_states[-1].mean(dim=1)
                else:
                    if isinstance(orig_outputs, tuple):
                        orig_feat = orig_outputs[0]
                    else:
                        orig_feat = orig_outputs

                    if orig_feat.dim() > 2:
                        orig_feat = orig_feat.mean(dim=list(range(1, orig_feat.dim() - 1)))
                        if orig_feat.dim() > 2:  # If still not 2D, flatten it
                            orig_feat = orig_feat.view(orig_feat.size(0), -1)

                # Get features for augmented image
                aug_outputs = self.model(aug_img)
                # Enhanced handling of different output types
                if hasattr(aug_outputs, 'pooler_output'):
                    aug_feat = aug_outputs.pooler_output
                elif hasattr(aug_outputs, 'last_hidden_state'):
                    aug_feat = aug_outputs.last_hidden_state[:, 0]
                elif hasattr(aug_outputs, 'logits') and not isinstance(aug_outputs, tuple):
                    aug_feat = aug_outputs.logits
                    if hasattr(aug_outputs, 'hidden_states') and aug_outputs.hidden_states is not None:
                        aug_feat = aug_outputs.hidden_states[-1][:, 0] if aug_feat.shape[1] == self.num_classes else \
                        aug_outputs.hidden_states[-1].mean(dim=1)
                else:
                    if isinstance(aug_outputs, tuple):
                        aug_feat = aug_outputs[0]
                    else:
                        aug_feat = aug_outputs

                    if aug_feat.dim() > 2:
                        aug_feat = aug_feat.mean(dim=list(range(1, aug_feat.dim() - 1)))
                        if aug_feat.dim() > 2:  # If still not 2D, flatten it
                            aug_feat = aug_feat.view(aug_feat.size(0), -1)

                # Compute similarity
                sim = F.cosine_similarity(orig_feat, aug_feat)
                similarities.append(sim)

        # Stack similarities and gather from all processes
        similarities = torch.stack(similarities)
        gathered_similarities = self.accelerator.gather(similarities)

        if self.accelerator.is_main_process:
            # Compute mean similarity
            robustness = gathered_similarities.mean().item()

            # Save histogram of similarities
            try:
                plt.figure(figsize=(10, 6))
                plt.hist(gathered_similarities.cpu().numpy(), bins=30, alpha=0.7)
                plt.axvline(x=robustness, color='r', linestyle='--',
                            label=f'Mean Similarity: {robustness:.4f}')
                plt.title('Feature Robustness to Augmentations')
                plt.xlabel('Cosine Similarity')
                plt.ylabel('Frequency')
                plt.legend()
                plt.savefig(os.path.join(self.output_dir, 'feature_robustness.png'))
                plt.close()
            except Exception as e:
                self.logger.warning(f"Failed to save robustness histogram: {str(e)}")
        else:
            robustness = 0.0

        # Broadcast result to all processes
        robustness_tensor = torch.tensor(robustness, device=self.accelerator.device)
        dist.broadcast(robustness_tensor, 0)
        robustness = robustness_tensor.item()

        return robustness
    

    def compute_evaluation_score(self):
        """Compute final evaluation score"""
        if self.accelerator.is_main_process:
            self.logger.info("Computing final evaluation score...")

        train_features, train_labels = self.extract_features(self.train_loader)
        val_features, val_labels = self.extract_features(self.val_loader)

        # First visualize embeddings with K-means (before linear probe)
        kmeans_vis_results = self.visualize_embeddings_kmeans(features=val_features,labels=val_labels)
        torch.cuda.empty_cache()
        print('kmeans_vis_results: ', kmeans_vis_results)
        # Get all metrics

        linear_metrics = self.linear_probe(train_features=train_features,train_labels=train_labels,val_features=val_features,val_labels=val_labels)

        print('linear_metrics: ', linear_metrics)
        torch.cuda.empty_cache()
        knn_metrics = self.knn_eval(train_features=train_features,train_labels=train_labels,val_features=val_features,val_labels=val_labels)
        torch.cuda.empty_cache()
        clustering_metrics = self.clustering_quality(features=val_features,labels=val_labels)
        torch.cuda.empty_cache() 
        robustness = self.feature_robustness()
        #else: 
        #    robustness=0
        # Define weights
        weights = {
            'linear_acc': 0.25,
            'knn_acc': 0.25,
            'clustering': 0.25,
            'robustness': 0.25
        }

        # Compute weighted score
        final_score = (
                weights['linear_acc'] * linear_metrics['accuracy'] +
                weights['knn_acc'] * knn_metrics['accuracy'] +
                weights['clustering'] * np.mean([
            clustering_metrics['silhouette'],
            clustering_metrics['ami']
        ]) +
                weights['robustness'] * robustness
        )

        # Save metrics summary
        if self.accelerator.is_main_process:
            try:
                # Create radar chart of metric components
                categories = ['Linear Acc', 'KNN Acc', 'Clustering AMI',
                            'Clustering Silhouette', 'Robustness']
                values = [
                    linear_metrics['accuracy'],
                    knn_metrics['accuracy'],
                    clustering_metrics['ami'],
                    clustering_metrics['silhouette'],
                    robustness
                ]

                # Ensure values are between 0 and 1 for plotting
                values = np.clip(values, 0, 1)

                # Create a simple bar chart instead of a radar chart to avoid dimension mismatch
                plt.figure(figsize=(10, 6))
                positions = np.arange(len(categories))
                plt.bar(positions, values)
                plt.xticks(positions, categories, rotation=45, ha='right')
                plt.ylim(0, max(values) * 1.2)  # Give some headroom above the bars
                plt.ylabel('Score')
                plt.title(f'Evaluation Metrics (Final Score: {final_score:.4f})')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'metrics_summary.png'))
                plt.close()
            except Exception as e:
                self.logger.warning(f"Failed to save metrics summary: {str(e)}")

        return {
            'model_architecture': self.model_type,
            'model_name': self.config.train.model_name,
            'final_score': final_score,
            'components': {
                'linear_probe': linear_metrics,
                'knn': knn_metrics,
                'clustering': clustering_metrics,
                'robustness': robustness,
                'kmeans_visualization': kmeans_vis_results if self.accelerator.is_main_process else None
            }
        }


    def evaluate(self):
        """Run complete evaluation pipeline"""
        if self.accelerator.is_main_process:
            self.logger.info("Starting comprehensive evaluation...")

        # Run all evaluations
        results = self.compute_evaluation_score()

        # Use accelerator's wait_for_everyone to sync processes
        self.accelerator.wait_for_everyone()

        # Print report on main process
        if self.accelerator.is_main_process:
            # The results are directly usable since we're on the main process
            final_score = results['final_score']
            linear_acc = results['components']['linear_probe']['accuracy']
            knn_acc = results['components']['knn']['accuracy']
            clustering_ami = results['components']['clustering']['ami']
            robustness = results['components']['robustness']
            kmeans_visualization = results['components']['kmeans_visualization']

            self.logger.info("\nEvaluation Report:")
            self.logger.info("-" * 50)
            self.logger.info(f"Final Score: {final_score:.4f}")
            self.logger.info("\nComponent Scores:")
            self.logger.info(f"Linear Probe Accuracy: {linear_acc:.4f}")
            self.logger.info(f"k-NN Accuracy: {knn_acc:.4f}")
            self.logger.info(f"Clustering AMI: {clustering_ami:.4f}")
            self.logger.info(f"Feature Robustness: {robustness:.4f}")
            
            if kmeans_visualization:
                self.logger.info("\nK-means Visualization Metrics:")
                self.logger.info(f"K-means Adjusted Mutual Info: {kmeans_visualization['adjusted_mutual_info']:.4f}")
                self.logger.info(f"K-means Silhouette Score: {kmeans_visualization['silhouette_score']:.4f}")
                self.logger.info(f"Visualization saved to: {os.path.join(self.output_dir, 'kmeans_embeddings_visualization.png')}")


            # Return results in the original format
            return results
        else:
            return None  # Non-main processes return None


def main():
    parser = argparse.ArgumentParser(description='Comprehensive feature evaluation')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file or config name') 

    args = parser.parse_args() 
    #output_dir
    # Set random seed
    set_seed(42)

    # Load configuration
    if os.path.isfile(args.config):
        # If a full path is provided    
        omgConf= OmegaConf.load(args.config)
        config=  OmegaConf.create(omgConf)  
    else:
        # If just a config name is provided (e.g., 'dino/main_config')
        config = load_config(args.config)
     
    # Add model name to output directory
    
    output_dir = config.train.main_output
    model_name = config.train.model_name
    output_dir = os.path.join(output_dir, model_name)
     
    os.makedirs(output_dir, exist_ok=True)
    # Initialize evaluator with config
    try:
        evaluator = ComprehensiveEvaluator(
            config=config,
            output_dir=output_dir
        )

        # Run evaluations
        results = evaluator.evaluate( )

        # Save results only on main process
        if evaluator.accelerator.is_main_process and results is not None:
            
            # Save results to JSON
            import json
            def convert_numpy_types(obj):
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(i) for i in obj]
                else:
                    return obj
                
            serializable_results = convert_numpy_types(results)
            
            # Save results to JSON
            import json
            try:
                results_path = os.path.join(output_dir, 'evaluation_results.json')
                with open(results_path, 'w') as f:
                    json.dump(serializable_results, f, indent=4) 
            except Exception as e: 
                import traceback
                print(traceback.format_exc())

    except Exception as e:
        # Get logger if available
        logger = logging.getLogger(__name__)
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    exit(main())