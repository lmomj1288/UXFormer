import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt
import cv2 
import os 
from operator import add
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

## Training ## 

def dice_loss_single(pred, target, smooth=1e-6):
    """
    pred: [B, 1, H, W] - logit values (Sigmoid not applied)
    target: [B, 1, H, W] - binary labels (0 or 1)
    """
    
    # Apply Sigmoid
    pred = torch.sigmoid(pred)  # [B, 1, H, W]
    
    # Reduce dimensions (remove channel dimension)
    pred = pred.squeeze(1)  # [B, H, W]
    target = target.squeeze(1)  # [B, H, W]
    
    # Calculate intersection and union
    intersection = (pred * target).sum(dim=(1, 2))  # [B]
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))  # [B]
    
    # Calculate dice loss
    dice = (2. * intersection + smooth) / (union + smooth)  # [B]
    loss = 1 - dice.mean()  # scalar
    
    return loss


def calculate_dice_score(pred, target, num_classes=None, smooth=1e-6):
    """
    pred: [B, C, H, W] - predicted values before activation function
    target: [B, C, H, W] - one-hot encoded labels (multi-class) or binary labels (single class)
    num_classes: number of classes (auto-detected if None)
    """
    
    if num_classes is None:
        num_classes = pred.size(1)
    
    # Apply Sigmoid for single class, Softmax for multi-class
    if num_classes == 1:
        pred = torch.sigmoid(pred)
    else:
        pred = F.softmax(pred, dim=1)
    
    dice_scores = []
    for class_idx in range(num_classes):
        # Process each class directly
        pred_class = pred[:, class_idx]  # [B, H, W]
        target_class = target[:, class_idx]  # [B, H, W]
        
        intersection = (pred_class * target_class).sum(dim=(1, 2))  # [B]
        union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))  # [B]
        
        dice = ((2. * intersection + smooth) / (union + smooth)).mean()  # scalar
        dice_scores.append(dice.item())
    
    # Return average Dice score across all classes
    return np.mean(dice_scores)

def combined_loss(pred, target, num_classes=1, dice_weight=0.9, ce_weight=0.1):
    # pred: [B, 1, H, W] - logits
    # target: [B, 1, H, W] - binary labels
    
    # Calculate Dice Loss
    dice = dice_loss_single(pred, target)
    
    # Calculate CrossEntropy Loss
    bce_loss = nn.BCEWithLogitsLoss()
    target_flat = target.squeeze(1)  # [B, H, W]
    pred_flat = pred.squeeze(1)  # [B, H, W]
    ce = bce_loss(pred_flat, target_flat)
    
    # Calculate total loss
    total_loss = dice_weight * dice + ce_weight * ce
    
    # Return each loss value separately for monitoring
    return total_loss, dice.item(), ce.item()


## Test ## 

def dice_coefficient(pred_mask, true_mask, smooth=1e-6):
    pred_flat = pred_mask.view(-1)
    target_flat = true_mask.view(-1)
    intersection = torch.sum(pred_flat * target_flat)
    return (2.0 * intersection + smooth) / (torch.sum(pred_flat) + torch.sum(target_flat) + smooth)

def calculate_dice_score(pred, target, smooth=1e-6):
    """
    Calculate Dice score for binary segmentation
    """
    # Apply Sigmoid to convert to probabilities
    pred = torch.sigmoid(pred)
    # Apply threshold to convert to binary mask
    pred_binary = (pred > 0.5).float()
    
    # Reduce dimensions (remove channel dimension)
    pred_binary = pred_binary.squeeze(1)  # (B, H, W)
    target = target.squeeze(1)  # (B, H, W)
    
    # Calculate Dice coefficient
    dice = dice_coefficient(pred_binary, target).item()
    
    return dice

def calculate_metrics(y_true, y_pred):
    """ 
    Calculate various segmentation metrics
    """
    # Convert ground truth
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    # Convert predictions
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    # Calculate metrics (with zero_division handling)
    score_jaccard = jaccard_score(y_true, y_pred, zero_division=0)
    score_f1 = f1_score(y_true, y_pred, zero_division=0)
    score_recall = recall_score(y_true, y_pred, zero_division=0)
    score_precision = precision_score(y_true, y_pred, zero_division=0)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]


def visualize_attention_map(att_map, original_image, alpha=0.5):
    """
    Function to visualize attention map by blending with the original image
    """
    # Ensure att_map is in the range [0, 1]
    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
    
    # Apply colormap to attention map
    heatmap = cv2.applyColorMap(np.uint8(255 * att_map), cv2.COLORMAP_JET)
    
    # Convert heatmap to RGB (from BGR)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match original image size if necessary
    if heatmap.shape[:2] != original_image.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Ensure original_image is in RGB
    if len(original_image.shape) == 2:  # If grayscale
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    elif len(original_image.shape) == 3 and original_image.shape[2] == 1:  # If grayscale with 3 dimensions
        original_image = cv2.cvtColor(original_image.squeeze(), cv2.COLOR_GRAY2RGB)
    
    # Normalize original image to 0-255 range if needed
    if original_image.max() <= 1.0:
        original_image = (original_image * 255).astype(np.uint8)
    
    # Blend original image with heatmap
    blended = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    
    return blended

def save_attention_maps(attention_maps, batch_idx, batch_size, output_dir, dataset, original_imgs=None):
    """
    Function to save attention maps as images - modified for UXFormer structure
    """
    # Create attention maps save directory
    attention_dir = os.path.join(output_dir, 'attention_maps')
    os.makedirs(attention_dir, exist_ok=True)
    
    # Create directory for each level (UXFormer returns decoder features)
    level_names = ['d4', 'd3', 'd2', 'd1']  # decoder outputs
    for level_name in level_names:
        level_dir = os.path.join(attention_dir, level_name)
        os.makedirs(level_dir, exist_ok=True)
        
        # Also create overlay directory (if original images are available)
        if original_imgs is not None:
            overlay_dir = os.path.join(attention_dir, f'{level_name}_overlay')
            os.makedirs(overlay_dir, exist_ok=True)
    
    # Process each image in the batch
    batch_size_actual = attention_maps[0].size(0)
    
    for i in range(batch_size_actual):
        idx = batch_idx * batch_size + i
        
        if idx >= len(dataset):
            break
        
        filename = f"image_{idx:04d}"
        
        # Prepare original image (for overlay)
        if original_imgs is not None:
            original_img = original_imgs[i, 0].cpu().numpy()
            # Convert 0-1 range to 0-255
            if original_img.max() <= 1.0:
                original_img = (original_img * 255).astype(np.uint8)
            else:
                original_img = original_img.astype(np.uint8)
        
        # Process each decoder level
        for level_idx, attention_map in enumerate(attention_maps):
            level_name = level_names[level_idx]
            
            # Extract attention map (i-th image, average across all channels)
            attn = attention_map[i].cpu().numpy()  # (C, H, W)
            
            # Average over channel dimension to get 2D map
            if len(attn.shape) == 3:
                attn_2d = np.mean(attn, axis=0)  # (H, W)
            else:
                attn_2d = attn  # already 2D
            
            # Attention map save directory
            level_dir = os.path.join(attention_dir, level_name)
            
            # 1. Save grayscale attention map (normalized 0-255 range)
            attn_normalized = ((attn_2d - attn_2d.min()) / 
                             (attn_2d.max() - attn_2d.min() + 1e-8) * 255).astype(np.uint8)
            # Resize to original image size (224x224)
            attn_resized = cv2.resize(attn_normalized, (224, 224), interpolation=cv2.INTER_LINEAR)
            Image.fromarray(attn_resized).save(
                os.path.join(level_dir, f"{filename}_gray.png")
            )
            
            # 2. Save heatmap only (attention map with colormap applied)
            att_map_normalized = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
            att_map_resized = cv2.resize(att_map_normalized, (224, 224), interpolation=cv2.INTER_LINEAR)
            heatmap = cv2.applyColorMap(np.uint8(255 * att_map_resized), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            Image.fromarray(heatmap_rgb).save(
                os.path.join(level_dir, f"{filename}_heatmap.png")
            )
            
            # 3. Save overlay version with original image (if original image is provided)
            if original_imgs is not None:
                try:
                    # Use visualize_attention_map function to create clean overlay
                    blended_image = visualize_attention_map(
                        att_map_resized,  # resized attention map
                        original_img,     # original image
                        alpha=0.5         # 50% transparency
                    )
                    
                    overlay_dir = os.path.join(attention_dir, f'{level_name}_overlay')
                    Image.fromarray(blended_image).save(
                        os.path.join(overlay_dir, f"{filename}_overlay.png")
                    )
                    
                    # Save with lighter transparency (alpha=0.3)
                    blended_light = visualize_attention_map(att_map_resized, original_img, alpha=0.3)
                    Image.fromarray(blended_light).save(
                        os.path.join(overlay_dir, f"{filename}_overlay_light.png")
                    )
                    
                    # Save with stronger transparency (alpha=0.7)
                    blended_strong = visualize_attention_map(att_map_resized, original_img, alpha=0.7)
                    Image.fromarray(blended_strong).save(
                        os.path.join(overlay_dir, f"{filename}_overlay_strong.png")
                    )
                except Exception as e:
                    print(f"Error creating overlay (level {level_name}, image {idx}): {e}")

def visualize_attention_statistics(attention_maps, output_dir):
    """
    Visualize and save statistical information of attention maps
    """
    stats_dir = os.path.join(output_dir, 'attention_stats')
    os.makedirs(stats_dir, exist_ok=True)
    
    # UXFormer returns 4 decoder outputs
    level_names = ['d4', 'd3', 'd2', 'd1']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Decoder Features Statistics', fontsize=16)
    
    for level, attention_map in enumerate(attention_maps):
        attn = attention_map.cpu().numpy()
        
        # Calculate mean values
        mean_values = np.mean(attn, axis=(2, 3))  # (B, C)
        mean_across_batch = np.mean(mean_values, axis=0)  # (C,)
        
        # Calculate standard deviation
        std_values = np.std(attn, axis=(2, 3))  # (B, C)
        std_across_batch = np.mean(std_values, axis=0)  # (C,)
        
        row = level // 2
        col = level % 2
        
        # Plot mean values per channel
        x_pos = np.arange(len(mean_across_batch))
        width = 0.35
        
        axes[row, col].bar(x_pos - width/2, mean_across_batch, width, 
                          alpha=0.7, label='Mean')
        axes[row, col].bar(x_pos + width/2, std_across_batch, width,
                          alpha=0.7, label='Std')
        axes[row, col].set_title(f'{level_names[level]} - Shape: {attn.shape}')
        axes[row, col].set_xlabel('Channel')
        axes[row, col].set_ylabel('Value')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, 'decoder_features_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save as text file
    with open(os.path.join(stats_dir, 'decoder_features_info.txt'), 'w') as f:
        f.write("Decoder Features Information\n")
        f.write("=" * 50 + "\n\n")
        
        for level, attention_map in enumerate(attention_maps):
            attn = attention_map.cpu().numpy()
            f.write(f"{level_names[level]}:\n")
            f.write(f"  Shape: {attn.shape}\n")
            f.write(f"  Min: {attn.min():.6f}\n")
            f.write(f"  Max: {attn.max():.6f}\n")
            f.write(f"  Mean: {attn.mean():.6f}\n")
            f.write(f"  Std: {attn.std():.6f}\n")
            f.write("\n")

def save_segmentation_results(original_imgs, filtered_imgs, targets, outputs, batch_idx, batch_size, output_dir, dataset):
    """
    Function to save segmentation results as images
    """
    # Create directories if they don't exist
    os.makedirs(os.path.join(output_dir, 'original'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'filtered'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'target'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'prediction'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'overlay'), exist_ok=True)
    
    # Convert prediction results to binary mask
    pred_probs = torch.sigmoid(outputs)  # (B, 1, H, W)
    pred_binary = (pred_probs > 0.5).float().cpu().numpy()  # (B, 1, H, W)
    
    # Process images in batch
    for i in range(original_imgs.size(0)):
        idx = batch_idx * batch_size + i
        
        if idx >= len(dataset):
            break
        
        # Original image
        original_img = original_imgs[i, 0].cpu().numpy()
        if original_img.max() <= 1.0:
            original_img = (original_img * 255).astype(np.uint8)
        else:
            original_img = original_img.astype(np.uint8)
        
        # Filtered image
        filtered_img = filtered_imgs[i, 0].cpu().numpy()
        if filtered_img.max() <= 1.0:
            filtered_img = (filtered_img * 255).astype(np.uint8)
        else:
            filtered_img = filtered_img.astype(np.uint8)
        
        # Target mask
        target_mask = targets[i, 0].cpu().numpy()
        if target_mask.max() <= 1.0:
            target_mask = (target_mask * 255).astype(np.uint8)
        else:
            target_mask = target_mask.astype(np.uint8)

        # Prediction mask
        pred_mask = pred_binary[i, 0]
        pred_mask = (pred_mask * 255).astype(np.uint8)
        
        # Create overlay image (original image + prediction mask + target mask)
        # Convert original image to 3-channel RGB
        original_rgb = np.stack([original_img] * 3, axis=2)
        
        # Apply mask threshold
        mask_pred = pred_mask > 127    # prediction mask (binary)
        mask_target = target_mask > 127  # target mask (binary)
        
        # Copy original image
        overlay = original_rgb.copy()
        
        # Display target mask in green (GT - Ground Truth)
        overlay[mask_target, 1] = 255  # green (target)
        
        # Display prediction mask in red
        overlay[mask_pred, 0] = 255  # red (prediction)
        
        # Save images
        filename = f"image_{idx:04d}"
        Image.fromarray(original_img).save(os.path.join(output_dir, 'original', f"{filename}.png"))
        Image.fromarray(filtered_img).save(os.path.join(output_dir, 'filtered', f"{filename}.png"))
        Image.fromarray(target_mask).save(os.path.join(output_dir, 'target', f"{filename}.png"))
        Image.fromarray(pred_mask).save(os.path.join(output_dir, 'prediction', f"{filename}.png"))
        Image.fromarray(overlay).save(os.path.join(output_dir, 'overlay', f"{filename}.png"))

def save_segmentation_results_with_attention(original_imgs, filtered_imgs, targets, outputs, attention_maps, 
                                           batch_idx, batch_size, output_dir, dataset):
    """
    Extended version of segmentation results save function that also saves attention maps
    """
    # Save existing segmentation results
    save_segmentation_results(original_imgs, filtered_imgs, targets, outputs, 
                            batch_idx, batch_size, output_dir, dataset)
    
    # Save attention maps
    save_attention_maps(attention_maps, batch_idx, batch_size, output_dir, dataset, original_imgs)

def evaluate_test_set_with_attention(model, test_loader, device, save_segmentation=False, 
                                   save_attention=False, output_dir=None, dataset=None):
    """
    Evaluation function modified for UXFormer model
    """
    model.eval()
    dice_scores = []
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    all_attention_maps = []  # storage for attention maps from all batches
    
    if (save_segmentation or save_attention) and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting evaluation... Total batches: {len(test_loader)}")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader)):
            original_imgs, filtered_imgs, targets = batch_data
            
            original_imgs = original_imgs.to(device)
            filtered_imgs = filtered_imgs.to(device)
            targets = targets.to(device)
            
            try:
                # Pass two inputs to UXFormer model
                # UXFormer returns final_output and [d4, d3, d2, d1]
                outputs, decoder_features = model(filtered_imgs, original_imgs)
                
                # Use decoder features as attention maps
                attention_maps = decoder_features
                
                # Collect attention maps (for statistics)
                if save_attention:
                    all_attention_maps.append([attn.clone() for attn in attention_maps])
                
                # Calculate Dice score
                dice_score = calculate_dice_score(outputs, targets)
                dice_scores.append(dice_score)
                
                # Calculate additional metrics
                for j in range(original_imgs.size(0)):
                    pred_j = torch.sigmoid(outputs[j])
                    score = calculate_metrics(targets[j], pred_j)
                    metrics_score = list(map(add, metrics_score, score))
                
                # Save segmentation results and attention maps
                if (save_segmentation or save_attention) and output_dir is not None:
                    if save_attention:
                        save_segmentation_results_with_attention(
                            original_imgs, filtered_imgs, targets, outputs, attention_maps,
                            batch_idx, test_loader.batch_size, output_dir, dataset
                        )
                    else:
                        save_segmentation_results(
                            original_imgs, filtered_imgs, targets, outputs,
                            batch_idx, test_loader.batch_size, output_dir, dataset
                        )
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                if batch_idx == 0:
                    raise
    
    # Save attention maps statistics
    if save_attention and output_dir is not None and all_attention_maps:
        print("Generating decoder features statistics...")
        # Use first batch's decoder features to generate statistics
        visualize_attention_statistics(all_attention_maps[0], output_dir)
    
    # Calculate average metrics
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    jaccard = metrics_score[0] / len(dataset)
    f1 = metrics_score[1] / len(dataset)
    recall = metrics_score[2] / len(dataset)
    precision = metrics_score[3] / len(dataset)
    acc = metrics_score[4] / len(dataset)
    
    # Save metrics scores to file
    if output_dir is not None:
        with open(os.path.join(output_dir, 'metrics_scores.txt'), 'w') as f:
            f.write(f"Average Dice Score: {avg_dice:.4f}\n")
            f.write(f"Jaccard: {jaccard:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            
            f.write("\nIndividual Batch Dice Scores:\n")
            for i, score in enumerate(dice_scores):
                f.write(f"Batch {i}: {score:.4f}\n")
    
    return avg_dice, jaccard, f1, recall, precision, acc