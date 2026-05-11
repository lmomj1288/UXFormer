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
            # 원본, 필터링된 이미지, 레이블 디렉토리 경로
            original_dir = join(data_dir, 'original')
            filtered_dir = join(data_dir, 'filtered')
            label_dir = join(data_dir, 'label')
            
            if not os.path.exists(original_dir) or not os.path.exists(label_dir) or not os.path.exists(filtered_dir):
                raise RuntimeError(f"Original, filtered, or label directory not found in {data_dir}")
            
            valid_triplets = 0
            skipped_triplets = 0
            
            # 원본 이미지 파일 목록 가져오기
            for img_file in sorted(os.listdir(original_dir)):
                original_path = join(original_dir, img_file)
                filtered_path = join(filtered_dir, img_file)  # 같은 이름의 필터링된 이미지 파일
                label_path = join(label_dir, img_file)  # 같은 이름의 레이블 파일
                
                # 입력 이미지와 레이블 이미지가 모두 있는지 확인
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
        # 원본 이미지 로드 및 변환
        original_img = Image.open(self.original_paths[idx]).convert('L')  # 그레이스케일로 변환
        original_tensor = self.transform(original_img)  # ToTensor가 [0, 1] 범위로 정규화
        
        # 필터링된 이미지 로드 및 변환
        filtered_img = Image.open(self.filtered_paths[idx]).convert('L')  # 그레이스케일로 변환
        filtered_tensor = self.transform(filtered_img)  # ToTensor가 [0, 1] 범위로 정규화
        
        # 레이블 이미지 로드 및 변환
        label_img = Image.open(self.label_paths[idx]).convert('L')  # 그레이스케일로 변환
        # 레이블은 ToTensor 변환 후 이진화 (0과 1로)
        label_tensor = self.transform(label_img)
        label_tensor = (label_tensor > 0.5).float()  # 임계값 0.5 이상은 1, 미만은 0
            
        return original_tensor, filtered_tensor, label_tensor