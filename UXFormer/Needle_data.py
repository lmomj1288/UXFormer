import torch
from torch.utils.data import Dataset
import os
from os.path import join, basename
import numpy as np
from PIL import Image
from torchvision import transforms

class NeedleDataset(Dataset):
    def __init__(self, data_dir, img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        
        self.original_paths = []
        self.filtered_paths = []
        self.label_paths = []
        
        if not os.path.exists(data_dir):
            raise RuntimeError(f"Directory not found: {data_dir}")
        
        try:
            original_dir = join(data_dir, 'original')
            filtered_dir = join(data_dir, 'filtered')
            label_dir = join(data_dir, 'label')
            
            if not os.path.exists(original_dir) or not os.path.exists(label_dir) or not os.path.exists(filtered_dir):
                raise RuntimeError(f"Original, filtered, or label directory not found in {data_dir}")
            
            valid_triplets = 0
            skipped_triplets = 0
            
            for img_file in sorted(os.listdir(original_dir)):
                original_path = join(original_dir, img_file)
                filtered_path = join(filtered_dir, img_file)  
                label_path = join(label_dir, img_file)  
                
                if os.path.exists(original_path) and os.path.exists(filtered_path) and os.path.exists(label_path):
                    self.original_paths.append(original_path)
                    self.filtered_paths.append(filtered_path)
                    self.label_paths.append(label_path)
                    valid_triplets += 1
                else:
                    missing = []
                    if not os.path.exists(original_path): missing.append("original")
                    if not os.path.exists(filtered_path): missing.append("filtered")
                    if not os.path.exists(label_path): missing.append("label")
                    print(f"Skipping {img_file}: Missing {', '.join(missing)} file(s)")
                    skipped_triplets += 1
            
            print(f"Found {valid_triplets} valid image triplets")
            print(f"Skipped {skipped_triplets} invalid triplets")
            
            if len(self.original_paths) == 0:
                raise RuntimeError(f"No valid image triplets found in {data_dir}")
                
        except Exception as e:
            print(f"Error during dataset initialization: {str(e)}")
            raise

    def __len__(self):
        return len(self.original_paths)

    def __getitem__(self, idx):
        original_img = Image.open(self.original_paths[idx]).convert('L')  
        original_tensor = self.transform(original_img)  
        
        filtered_img = Image.open(self.filtered_paths[idx]).convert('L')  
        filtered_tensor = self.transform(filtered_img) 
        
        label_img = Image.open(self.label_paths[idx]).convert('L') 
        
        label_tensor = self.transform(label_img)
        label_tensor = (label_tensor > 0.5).float() 
            
        return original_tensor, filtered_tensor, label_tensor
