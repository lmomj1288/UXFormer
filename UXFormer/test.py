import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from os.path import join
import torch.utils.data as D
from utils import * 
from Needle_data import NeedleDataset
from model import UXFormer

test_dir = './dataset/test'
model_path = './checkpoint/unified_lr_seg_model_best.pt'
gpu_id = 2
batch_size = 1
save_segmentation = True
save_attention = True  
output_dir = './result'
img_size = (224, 224)

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
if torch.cuda.is_available():
    device = torch.device("cuda:0")  
    print(f"Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

try:
    model = UXFormer(n_channels=1, n_channels_transformer=1, n_classes=1, n_filters=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False), strict=False)
    print(f"Model loaded successfully: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    test_dataset = NeedleDataset(test_dir, img_size=img_size)
    test_loader = D.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"Test dataset loaded: {len(test_dataset)} images")
except Exception as e:
    print(f"Error loading dataset: {e}")

try:
    avg_dice, jaccard, f1, recall, precision, acc = evaluate_test_set_with_attention(
        model, test_loader, device, save_segmentation, save_attention, output_dir, test_dataset
    )
    
    print("\n" + "="*70)
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Jaccard: {jaccard:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print("="*70)
    
    if save_segmentation or save_attention: 
        print(f"\nResults saved to: {output_dir}")
        if save_attention:
            print("- Decoder features saved in 'attention_maps' folder")
            print("- Grayscale, heatmap, and overlay versions saved per decoder level")
            print("- Decoder feature statistics saved in 'attention_stats' folder")
            
except Exception as e:
    print(f"Error during evaluation: {e}")
    import traceback
    traceback.print_exc()