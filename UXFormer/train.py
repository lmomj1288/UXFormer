import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torch.optim as optim
import numpy as np
import csv 
import os
from timeit import default_timer as timer
from os.path import join
from Needle_data import NeedleDataset
from model import UXFormer
from utils import * 

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

root_dir = './'
needle_data_dir = './dataset'
train_data_dir = join(needle_data_dir, 'train') 
test_data_dir = join(needle_data_dir, 'test')  
val_data_dir = join(needle_data_dir, 'val')   
losses_dir = join(root_dir, 'losses') 
models_dir = join(root_dir, 'saved_models')  
checkpoint_dir = join(root_dir, 'checkpoints')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1

train_dataset = NeedleDataset(train_data_dir)
train_loader = D.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = NeedleDataset(val_data_dir)
dev_loader = D.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

seg_model = UXFormer(n_channels=1, n_channels_transformer=1, n_classes=1, n_filters=32).to(device)

def get_parameters(model):
    params = []
    for name, param in model.named_parameters():
        params.append(param)
        print(f"Parameter: {name}")
    print(f"\nTotal parameters count: {len(params)}")
    return params

params = get_parameters(seg_model) 

optimizer = optim.AdamW(params, lr=0.0001)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
    verbose=True,
)

def train_epoch(seg_model, optimizer, train_loader, epoch, visualize_dir='./visualizations'):
    seg_model.train()
    total_loss = 0
    total_recon = 0
    total_dice_value = 0
    total_ce_value = 0
    recon_weight = 1

    for batch_idx, batch in enumerate(train_loader):
        original_img, filtered_img, seg = batch
        original_img = original_img.to(device)
        filtered_img = filtered_img.to(device)
        seg = seg.to(device)

        optimizer.zero_grad()

        output, _ = seg_model(filtered_img, original_img)
        
        dice_loss, dice_value, ce_value = combined_loss(output, seg)
        
        total_dice_value += dice_value
        total_ce_value += ce_value
        
        loss_d = torch.mean(dice_loss)
        loss = torch.mean(recon_weight * dice_loss)

        total_recon += loss_d.item()
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    avg_dice_value = total_dice_value / len(train_loader)
    avg_ce_value = total_ce_value / len(train_loader)   
    
    print(f'train_avg_dice_loss: {avg_dice_value:.4f}')
    print(f'train_avg_ce_loss: {avg_ce_value:.4f}')
    print(f'train_combine_loss: {total_recon/len(train_loader):.4f}')
    print()
    
    return total_loss / len(train_loader)

def evaluate(seg_model, dev_loader, epoch, save_dir, scheduler, visualize_dir='./visualizations_val'): 
    seg_model.eval()
    
    total_loss = 0
    total_recon = 0
    total_dice_value = 0
    total_ce_value = 0
    recon_weight = 1
    total_dice_score = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dev_loader):
            original_img, filtered_img, y = batch
            original_img = original_img.to(device)
            filtered_img = filtered_img.to(device)
            y = y.to(device)
            
            output, _ = seg_model(filtered_img, original_img)
            dice_loss, dice_value, ce_value = combined_loss(output, y)
            
            total_dice_value += dice_value
            total_ce_value += ce_value
            
            dice_score = calculate_dice_score(output, y)
            total_dice_score += dice_score 
            
            loss_d = torch.mean(dice_loss)
            loss = torch.mean(recon_weight * dice_loss)

            total_recon += loss_d.item()
            total_loss += loss.item()

    avg_loss = total_loss / len(dev_loader)
    avg_dice_score = total_dice_score / len(dev_loader)
    avg_dice_value = total_dice_value / len(dev_loader)
    avg_ce_value = total_ce_value / len(dev_loader)
        
    print(f'Epoch {epoch} Validation Metrics:')
    print(f'Average Dice Score: {avg_dice_score:.4f}')
    print()
    print(f'dev_avg_dice_loss: {avg_dice_value:.4f}')
    print(f'dev_avg_ce_loss: {avg_ce_value:.4f}')
    print(f'dev_combine_loss: {total_recon/len(dev_loader):.4f}')
    print()
    
    scheduler.step(avg_loss)

    return avg_loss, avg_dice_score

total_epochs = 300
train_losses = []; dev_losses = []
best_loss = float('inf')  

results_dir = './logs'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

csv_path = join(results_dir, 'training_metrics.csv')

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train_Loss', 'Dev_Loss', 'Dev_Dice'])

try:
    for epoch in range(1, total_epochs+1):
        start_time = timer()

        train_loss = train_epoch(seg_model, optimizer, train_loader, epoch)
        train_losses.append(train_loss)

        dev_loss, dev_dice = evaluate(seg_model, dev_loader, epoch, results_dir, scheduler)
        dev_losses.append(dev_loss)
        
        end_time = timer()
        epoch_time = end_time - start_time

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 
                f"{train_loss:.6f}", 
                f"{dev_loss:.6f}", 
                f"{dev_dice:.6f}"
            ])

        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save(seg_model.state_dict(), join(results_dir, 'unified_lr_seg_model_best.pt'))
            print(f"New best model saved with Low_loss_epoch_{epoch}: {best_loss:.4f}")

        if epoch % 10 == 0:
            torch.save(seg_model.state_dict(), join(results_dir, f'unified_lr_seg_model_epoch{epoch}.pt'))
        
        print(f"Epoch {epoch}: Train-loss: {train_loss:.4f}, "
              f"Valid-loss: {dev_loss:.4f}, "
              f"Valid-dice: {dev_dice:.4f}, "
              f"Epoch-time = {epoch_time:.3f}s")
        print()
    
    print('Training completed.')
    
except KeyboardInterrupt:
    print('Training interrupted.')
    print(f'Last saved metrics: Epoch {epoch}, Train-loss: {train_loss:.4f}, Valid-loss: {dev_loss:.4f}, Valid-dice: {dev_dice:.4f}')
    
    torch.save(seg_model.state_dict(), join(results_dir, f'unified_lr_seg_model_interrupted_epoch{epoch}.pt'))
    print(f'Interrupted model saved at epoch {epoch}')

print(f'Best validation loss: {best_loss:.4f}')
print(f'Results saved to: {results_dir}')