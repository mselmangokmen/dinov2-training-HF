



import inspect 
import numpy as np   
import torch 
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler 

import numpy as np  
import torch.nn.functional as F   
    

def get_attention_maps(norm_image, model, device, patch_size=16, num_register_tokens=0, img_shape=(512,512)):
    """
    Extracts self-attention maps from the last layer of a model.
    
    Returns:
        - cls_to_patch_maps: Attention from CLS to Patch tokens
        - reg_to_patch_maps: Attention from Register tokens to Patch tokens (if registers exist)
    """
    bs = norm_image.shape[0]
    #print('num_register_tokens: ', num_register_tokens,'img_shape:',img_shape,'patch_size: ', patch_size )
    # Resize image to match patch size if needed
    h_img, w_img = norm_image.shape[-2:]
    target_h, target_w = (h_img // patch_size) * patch_size, (w_img // patch_size) * patch_size

    if img_shape != (norm_image.shape[-2], norm_image.shape[-1]):  
        target_h, target_w = img_shape
        
    if h_img != target_h or w_img != target_w:
        norm_image = F.interpolate(
            norm_image, 
            size=(target_h, target_w), 
            mode='bilinear', 
            align_corners=False, 
            antialias=True
        )
        h_img, w_img = target_h, target_w
    
    # Calculate feature map dimensions
    h_featmap, w_featmap = h_img // patch_size, w_img // patch_size
    expected_num_patch_tokens = h_featmap * w_featmap
    expected_total_tokens = 1 + num_register_tokens + expected_num_patch_tokens
    
    # Move image to device
    images = norm_image.to(device, dtype=torch.float32)
    raw_attentions = None
    
    # Try to extract attention maps - first with DINOv1 method
    if hasattr(model, 'get_last_selfattention') and callable(model.get_last_selfattention):
        raw_attentions = model.get_last_selfattention(images)
        if raw_attentions is not None:
            if raw_attentions.ndim != 4 or raw_attentions.shape[0] != bs or \
               raw_attentions.shape[2] != expected_total_tokens or \
               raw_attentions.shape[3] != expected_total_tokens:
                raw_attentions = None
    
    # If DINOv1 method failed, try general method
    if raw_attentions is None:
        forward_kwargs = {}
        try:
            sig = inspect.signature(model.forward)
            if 'output_attentions' in sig.parameters:
                forward_kwargs['output_attentions'] = True
            elif 'return_attention' in sig.parameters:
                forward_kwargs['return_attention'] = True
            if 'interpolate_pos_encoding' in sig.parameters:
                forward_kwargs['interpolate_pos_encoding'] = True
            outputs = model(images, **forward_kwargs)
            
            # Search for attention in outputs
            if hasattr(outputs, 'attentions') and outputs.attentions is not None and len(outputs.attentions) > 0:
                raw_attentions = outputs.attentions[-1]
            elif isinstance(outputs, (tuple, list)):
                for item in reversed(outputs):
                    if isinstance(item, (list, tuple)) and len(item) > 0 and all(isinstance(att, torch.Tensor) for att in item):
                        potential_att = item[-1]
                        if potential_att.ndim == 4 and potential_att.shape[0] == bs and potential_att.shape[2] == potential_att.shape[3]:
                            raw_attentions = potential_att
                            break
                    elif isinstance(item, torch.Tensor) and item.ndim == 4 and item.shape[0] == bs and item.shape[2] == item.shape[3]:
                        raw_attentions = item
                        break
            elif isinstance(outputs, torch.Tensor) and outputs.ndim == 4 and outputs.shape[0] == bs and outputs.shape[2] == outputs.shape[3]:
                raw_attentions = outputs
                
            if raw_attentions is None:
                raise ValueError("No recognizable attention maps found in model output.")
                
        except Exception as e:
            raise ValueError(f"Could not extract attention maps: {e}")
    
    # Validate attention maps
    if raw_attentions is None:
        raise ValueError("Could not extract attention maps with any method.")
    if raw_attentions.ndim != 4 or raw_attentions.shape[0] != bs or raw_attentions.shape[2] != raw_attentions.shape[3] or raw_attentions.shape[2] != expected_total_tokens:
        raise ValueError(f"Attention tensor shape mismatch. Expected {(bs, 'nh', expected_total_tokens, expected_total_tokens)}, got {raw_attentions.shape}.")
    
    nh = raw_attentions.shape[1]  # Number of attention heads
    
    # Calculate indices for different token types
    patch_token_start_index = 1 + num_register_tokens
    patch_token_end_index = patch_token_start_index + expected_num_patch_tokens
    
    # Extract CLS -> Patch attention maps
    cls_to_patch_attentions = raw_attentions[:, :, 0, patch_token_start_index:patch_token_end_index]
    
    if expected_num_patch_tokens > 0:
        try:
            cls_to_patch_maps = cls_to_patch_attentions.reshape(bs, nh, h_featmap, w_featmap).cpu().detach().numpy()
        except RuntimeError as e:
            raise RuntimeError(f"Failed to reshape cls->patch attention: {e}")
    else:
        cls_to_patch_maps = np.zeros((bs, nh, h_featmap, w_featmap))
    
    # Initialize register-related output
    reg_to_patch_maps = None
    
    # Process register tokens if they exist
    if num_register_tokens > 0:
        register_token_start_index = 1
        register_token_end_index = 1 + num_register_tokens
        
        # Calculate Register -> Patch attention maps
        if expected_num_patch_tokens > 0:
            reg_to_patch_attentions = raw_attentions[
                :, :, register_token_start_index:register_token_end_index, patch_token_start_index:patch_token_end_index
            ]
            
            try: 
                #print('reg_to_patch_attentions shape: ', reg_to_patch_attentions.shape)
                reshaped = reg_to_patch_attentions.reshape(bs, nh, num_register_tokens, h_featmap, w_featmap)
                averaged = reshaped.mean(dim=1)
                permuted = averaged.permute(0, 1, 3, 2)
                reg_to_patch_maps = permuted.cpu().detach().numpy()
                 
                for i in range(num_register_tokens): 
                    reg_to_patch_maps[0,i,:,:] = reg_to_patch_maps[0, i, :, :].T
            except RuntimeError as e:
                raise RuntimeError(f"Failed to reshape reg->patch attention: {e}")
        else:

            reg_to_patch_maps=None
    return cls_to_patch_maps, reg_to_patch_maps, (target_h, target_w)

def normalize_attention(attentions):
    
    normalized_attentions = []
    for head_idx in range(attentions.shape[0]):
        attention = attentions[head_idx, :, :]
        attention_min = attention.min()
        attention_max = attention.max()
        normalized_attention = (attention - attention_min) / (attention_max - attention_min)
        normalized_attentions.append(normalized_attention) 
    return np.array(normalized_attentions)
 
def generate_pca_from_attention_map( attentions, patch_size=8,n_components=3,img_shape=(512,512),  nth_head=None ,
                normalize_pca=False ):   
  
    #attentions=normalize_attention(attentions)
    if nth_head: 
        attentions= attentions[:,:nth_head,:,:] 

    attentions=np.squeeze(attentions)
    nh = attentions.shape[0]     

    attentions = attentions.reshape(nh, -1) 
 
    pca = PCA(n_components=n_components, whiten=True)
    pca_result = pca.fit_transform(attentions.T)   
    pca_result = MinMaxScaler().fit_transform(pca_result)  
     
    #print('pca_result max and min: ', np.amax(pca_result), np.amin(pca_result))
    w_featmap, h_featmap = img_shape[0] // patch_size, img_shape[1] // patch_size 
    pca_image = pca_result.reshape(w_featmap, h_featmap,n_components)
    
    pca_channels = [pca_image[:, :, i] for i in range(n_components)]  
    pca_channels=np.array(pca_channels)  
    #print('pca_channels shape: ', pca_channels.shape)
    #filtered_channels= filter_pca(pca_channels=pca_channels) 
    filtered_channels= pca_channels
    gauss_combined_pca = np.sum(filtered_channels, axis=0)  
    return filtered_channels 
 