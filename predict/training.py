import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import seaborn as sns

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', clip_len=16, transforms=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transforms = transforms
        
        folder_path = os.path.join(root_dir, split)
        self.classes = sorted(os.listdir(folder_path))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.fnames, self.labels = [], []
        for label in self.classes:
            class_path = os.path.join(folder_path, label)
            for fname in os.listdir(class_path):
                if fname.endswith(('.mp4', '.webm', '.avi')):
                    self.fnames.append(os.path.join(class_path, fname))
                    self.labels.append(self.class_to_idx[label])
                
        print(f'Found {len(self.fnames)} videos for {split} belonging to {len(self.classes)} classes')
                
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, index):
        try:
            import torchvision.io as io
            video_path = self.fnames[index]
            video_frames, _, _ = io.read_video(video_path, pts_unit='sec')
            
            # Handle video frames
            if video_frames.shape[0] < self.clip_len:
                video_frames = torch.cat([video_frames] * (self.clip_len // video_frames.shape[0] + 1), dim=0)
            
            total_frames = video_frames.shape[0]
            start_idx = np.random.randint(0, total_frames - self.clip_len) if total_frames > self.clip_len else 0
            video_clip = video_frames[start_idx:start_idx + self.clip_len]
            
            # Ensure exact clip length
            if video_clip.shape[0] < self.clip_len:
                video_clip = torch.cat([video_clip, video_clip[-1].unsqueeze(0).repeat(self.clip_len - video_clip.shape[0], 1, 1, 1)], dim=0)
            elif video_clip.shape[0] > self.clip_len:
                video_clip = video_clip[:self.clip_len]
            
            if self.transforms:
                transformed_frames = []
                for frame in video_clip:
                    frame = frame.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
                    frame = self.transforms(frame)
                    transformed_frames.append(frame)
                video_tensor = torch.stack(transformed_frames, dim=1)  # [C, T, H, W]
                return video_tensor, self.labels[index]
            
            # Without transforms
            video_clip = video_clip.permute(3, 0, 1, 2)  # [T, H, W, C] -> [C, T, H, W]
            return video_clip.float() / 255.0, self.labels[index]
        
        except Exception as e:
            print(f"Error loading video {self.fnames[index]}: {e}")
            return torch.zeros(3, self.clip_len, 224, 224), self.labels[index]

def get_model(num_classes, pretrained=True):
    try:
        from torchvision.models.video import r3d_18
        model = r3d_18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    except (ImportError, AttributeError):
        from torchvision.models import resnet18
        model = resnet18(pretrained=pretrained)
        model.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

def train_model(model, dataloaders, test_loader, criterion, optimizer, scheduler=None, num_epochs=25, patience=5, device='cuda'):
    model = model.to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'early stop {patience_counter}/{patience}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            y_true, y_pred = [], []
            
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = accuracy_score(y_true, y_pred)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
                if scheduler: scheduler.step()
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs!')
                    model.load_state_dict(best_model_wts)
                    history['total_training_time'] = time.time() - start_time
                    return model, history
        
        # 每 5 個 epoch 在測試集上評估
        # if (epoch + 1) % 5 == 0:
        #     print(f"\nEvaluating model on test dataset at epoch {epoch+1}...")
        #     acc1, _, _ = evaluate_model(model, test_loader, device, run_id=f"epoch_{epoch+1}")
            
        #     elapsed_time = time.time() - start_time
        #     hours, remainder = divmod(elapsed_time, 3600)
        #     minutes, seconds = divmod(remainder, 60)
            
        #     time_str = ""
        #     if hours > 0:
        #         time_str += f"{int(hours)}h "
        #     if minutes > 0 or hours > 0:
        #         time_str += f"{int(minutes)}mins"
        #     time_str += f"{int(seconds)}s"
            
        #     print(f"epoch {epoch+1} test accuracy: {acc1:.4f} training time: {time_str}\n")
    
    # history['total_training_time'] = time.time() - start_time
    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_model(model, test_loader, device='cuda', run_id=''):
    model.eval()
    y_true, y_pred = [], []
    top5_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # ACC1 (Top-1)
            _, preds = torch.max(outputs, 1)
            
            # ACC5 (Top-5)
            _, top5_preds = torch.topk(outputs, k=5, dim=1)
            batch_size = labels.size(0)
            for i in range(batch_size):
                if labels[i] in top5_preds[i]:
                    top5_correct += 1
            total_samples += batch_size
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # Calculate metrics
    acc1 = accuracy_score(y_true, y_pred)
    acc5 = top5_correct / total_samples
    report = classification_report(y_true, y_pred, target_names=test_loader.dataset.classes)
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=test_loader.dataset.classes, 
                yticklabels=test_loader.dataset.classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{run_id}.png')
    plt.close()
    
    # Plot accuracy comparison
    plt.figure(figsize=(8, 6))
    accuracies = [acc1*100, acc5*100]
    labels = ['ACC1 (Top-1)', 'ACC5 (Top-5)']
    bars = plt.bar(labels, accuracies, color=['royalblue', 'forestgreen'], width=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, 100)
    plt.title('Top-1 vs Top-5 Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.tight_layout()
    plt.savefig(f'accuracy_comparison_{run_id}.png')
    plt.close()
    
    return acc1, acc5, report

def plot_training_history(history, run_id=''):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if 'total_training_time' in history:
        total_time = history['total_training_time']
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        plt.figtext(0.5, 0.01, f'Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s', 
                    ha='center', fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.savefig(f'training_history_{run_id}.png')
    plt.close()

def main():
    # Configuration
    run_id = 'run0115_finetune_EDA'
    data_root = r"dataset3/2026/dataset"
    batch_size = 4
    num_epochs = 300
    early_stopping_patience = 10
    learning_rate = 0.0001
    clip_length = 16
    
    total_start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import seaborn as sns

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', clip_len=16, transforms=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transforms = transforms
        
        folder_path = os.path.join(root_dir, split)
        self.classes = sorted(os.listdir(folder_path))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.fnames, self.labels = [], []
        for label in self.classes:
            class_path = os.path.join(folder_path, label)
            for fname in os.listdir(class_path):
                if fname.endswith(('.mp4', '.webm', '.avi')):
                    self.fnames.append(os.path.join(class_path, fname))
                    self.labels.append(self.class_to_idx[label])
                
        print(f'Found {len(self.fnames)} videos for {split} belonging to {len(self.classes)} classes')
                
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, index):
        try:
            import torchvision.io as io
            video_path = self.fnames[index]
            video_frames, _, _ = io.read_video(video_path, pts_unit='sec')
            
            # Handle video frames
            if video_frames.shape[0] < self.clip_len:
                video_frames = torch.cat([video_frames] * (self.clip_len // video_frames.shape[0] + 1), dim=0)
            
            total_frames = video_frames.shape[0]
            start_idx = np.random.randint(0, total_frames - self.clip_len) if total_frames > self.clip_len else 0
            video_clip = video_frames[start_idx:start_idx + self.clip_len]
            
            # Ensure exact clip length
            if video_clip.shape[0] < self.clip_len:
                video_clip = torch.cat([video_clip, video_clip[-1].unsqueeze(0).repeat(self.clip_len - video_clip.shape[0], 1, 1, 1)], dim=0)
            elif video_clip.shape[0] > self.clip_len:
                video_clip = video_clip[:self.clip_len]
            
            if self.transforms:
                transformed_frames = []
                for frame in video_clip:
                    frame = frame.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
                    frame = self.transforms(frame)
                    transformed_frames.append(frame)
                video_tensor = torch.stack(transformed_frames, dim=1)  # [C, T, H, W]
                return video_tensor, self.labels[index]
            
            # Without transforms
            video_clip = video_clip.permute(3, 0, 1, 2)  # [T, H, W, C] -> [C, T, H, W]
            return video_clip.float() / 255.0, self.labels[index]
        
        except Exception as e:
            print(f"Error loading video {self.fnames[index]}: {e}")
            return torch.zeros(3, self.clip_len, 224, 224), self.labels[index]

def get_model(num_classes, pretrained=True):
    try:
        from torchvision.models.video import r3d_18
        model = r3d_18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    except (ImportError, AttributeError):
        from torchvision.models import resnet18
        model = resnet18(pretrained=pretrained)
        model.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

def train_model(model, dataloaders, test_loader, criterion, optimizer, scheduler=None, num_epochs=25, patience=5, device='cuda'):
    model = model.to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'early stop {patience_counter}/{patience}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            y_true, y_pred = [], []
            
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = accuracy_score(y_true, y_pred)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
                if scheduler: scheduler.step()
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs!')
                    model.load_state_dict(best_model_wts)
                    history['total_training_time'] = time.time() - start_time
                    return model, history
        
        # 每 5 個 epoch 在測試集上評估
        # if (epoch + 1) % 5 == 0:
        #     print(f"\nEvaluating model on test dataset at epoch {epoch+1}...")
        #     acc1, _, _ = evaluate_model(model, test_loader, device, run_id=f"epoch_{epoch+1}")
            
        #     elapsed_time = time.time() - start_time
        #     hours, remainder = divmod(elapsed_time, 3600)
        #     minutes, seconds = divmod(remainder, 60)
            
        #     time_str = ""
        #     if hours > 0:
        #         time_str += f"{int(hours)}h "
        #     if minutes > 0 or hours > 0:
        #         time_str += f"{int(minutes)}mins"
        #     time_str += f"{int(seconds)}s"
            
        #     print(f"epoch {epoch+1} test accuracy: {acc1:.4f} training time: {time_str}\n")
    
    # history['total_training_time'] = time.time() - start_time
    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_model(model, test_loader, device='cuda', run_id=''):
    model.eval()
    y_true, y_pred = [], []
    top5_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # ACC1 (Top-1)
            _, preds = torch.max(outputs, 1)
            
            # ACC5 (Top-5)
            _, top5_preds = torch.topk(outputs, k=5, dim=1)
            batch_size = labels.size(0)
            for i in range(batch_size):
                if labels[i] in top5_preds[i]:
                    top5_correct += 1
            total_samples += batch_size
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # Calculate metrics
    acc1 = accuracy_score(y_true, y_pred)
    acc5 = top5_correct / total_samples
    report = classification_report(y_true, y_pred, target_names=test_loader.dataset.classes)
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=test_loader.dataset.classes, 
                yticklabels=test_loader.dataset.classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{run_id}.png')
    plt.close()
    
    # Plot accuracy comparison
    plt.figure(figsize=(8, 6))
    accuracies = [acc1*100, acc5*100]
    labels = ['ACC1 (Top-1)', 'ACC5 (Top-5)']
    bars = plt.bar(labels, accuracies, color=['royalblue', 'forestgreen'], width=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, 100)
    plt.title('Top-1 vs Top-5 Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.tight_layout()
    plt.savefig(f'accuracy_comparison_{run_id}.png')
    plt.close()
    
    return acc1, acc5, report

def plot_training_history(history, run_id=''):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if 'total_training_time' in history:
        total_time = history['total_training_time']
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        plt.figtext(0.5, 0.01, f'Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s', 
                    ha='center', fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.savefig(f'training_history_{run_id}.png')
    plt.close()

def main():
    # Configuration
    run_id = 'run0115_finetune_EDA'
    data_root = r"dataset3/2026/dataset"
    batch_size = 4
    num_epochs = 300
    early_stopping_patience = 10
    learning_rate = 0.0001
    clip_length = 16
    
    total_start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    train_dataset = VideoDataset(data_root, split='train', clip_len=clip_length, transforms=transform)
    val_dataset = VideoDataset(data_root, split='val', clip_len=clip_length, transforms=transform)
    test_dataset = VideoDataset(data_root, split='test', clip_len=clip_length, transforms=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    # Create model
    num_classes = len(train_dataset.classes)
    print(f'Number of classes: {num_classes}')
    model = get_model(num_classes=num_classes, pretrained=True)
    
    # Freeze most layers, only train last few
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'layer4' or 'fc' in name:  #如果
            param.requires_grad = True 
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train model
    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        test_loader=test_loader,  # 傳入 test_loader
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        patience=early_stopping_patience,
        device=device
    )
    
    # Evaluate and visualize results
    print("\nEvaluating model on test dataset...")
    acc1, acc5, test_report = evaluate_model(model, test_loader, device, run_id)
    
    print(f"\nACC1 (Top-1 Accuracy): {acc1:.4f} ({acc1*100:.2f}%)")
    print(f"ACC5 (Top-5 Accuracy): {acc5:.4f} ({acc5*100:.2f}%)")
    print("\nDetailed Classification Report:")
    print(test_report)
    
    # Save results and model
    with open(f'test_results_{run_id}.txt', 'w') as f:
        f.write(f"ACC1 (Top-1 Accuracy): {acc1:.4f} ({acc1*100:.2f}%)\n")
        f.write(f"ACC5 (Top-5 Accuracy): {acc5:.4f} ({acc5*100:.2f}%)\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(test_report)
    
    plot_training_history(history, run_id)
    torch.save(model.state_dict(), f'{run_id}.pth')
    
    # Total execution time
    # total_time = time.time() - total_start_time
    # hours, remainder = divmod(total_time, 3600)
    # minutes, seconds = divmod(remainder, 60)
    # print(f'Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s')
    print('Model saved!')
    print(f'{run_id} is done~~~~~~~~~~~~~~~~~~~~~')

if __name__ == '__main__':
    main()








# import os
# import gc
# import time
# import torch
# import platform
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import torch.nn as nn
# import torch.optim as optim
# from collections import defaultdict
# from torchvision.io import read_video
# from torch.utils.data import DataLoader
# from torchvision import models, transforms
# from pytorchvideo.models.hub import slowfast_r50
# from torchvision.models.video import R3D_18_Weights
# from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# import sys
# sys.path.append('D:/GAN/pytorch-i3d')
# from pytorch_i3d import InceptionI3d

# # 啟用 cudnn benchmark 加速
# torch.backends.cudnn.benchmark = True

# # ============================================================================
# # VideoDataset類別（支援MP4）
# # ============================================================================
# class VideoDataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir, split='train', clip_len=64, transform=None):
#         """
#         支援MP4影片的資料集
#         資料夾結構: root_dir/split/class_name/video.mp4
#         """
#         self.root_dir = os.path.join(root_dir, split)
#         self.clip_len = clip_len
#         self.transform = transform
#         self.classes = sorted(os.listdir(self.root_dir))
#         self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
#         self.videos = []
        
#         for cls_name in self.classes:
#             cls_dir = os.path.join(self.root_dir, cls_name)
#             if not os.path.isdir(cls_dir):
#                 continue
#             for video_name in os.listdir(cls_dir):
#                 if video_name.endswith('.mp4'):
#                     self.videos.append((os.path.join(cls_dir, video_name), cls_name))
        
#         print(f"  {split.upper()}: {len(self.videos)} videos")

#     def __len__(self):
#         return len(self.videos)

#     def __getitem__(self, idx):
#         video_path, cls_name = self.videos[idx]
#         label = self.class_to_idx[cls_name]

#         try:
#             video, _, _ = read_video(video_path, pts_unit='sec')
#             total_frames = video.shape[0]
#         except Exception as e:
#             print(f"Error reading {video_path}: {e}")
#             return None, None

#         if total_frames == 0:
#             print(f"Warning: {video_path} has 0 frames")
#             return None, None

#         # Clip長度處理
#         if total_frames < self.clip_len:
#             pad_len = self.clip_len - total_frames
#             last_frame = video[-1:].repeat(pad_len, 1, 1, 1)
#             video = torch.cat([video, last_frame], dim=0)
#         else:
#             start_idx = np.random.randint(0, total_frames - self.clip_len + 1)
#             video = video[start_idx:start_idx + self.clip_len]

#         # (T, H, W, C) → (C, T, H, W)
#         video = video.permute(3, 0, 1, 2).float() / 255.0

#         if self.transform:
#             video = torch.stack([self.transform(video[:, i]) for i in range(video.shape[1])], dim=1)

#         return video, label


# # ============================================================================
# # 輔助函數
# # ============================================================================
# def top_k_accuracy(output, target, k=1):
#     with torch.no_grad():
#         _, pred = output.topk(k, dim=1, largest=True, sorted=True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#         correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#         return correct_k.mul_(100.0 / output.size(0)).item()


# def prepare_inputs(model_name, videos):
#     """根據論文附錄準備輸入"""
#     if model_name == 'slowfast_r50':
#         fast_pathway = videos[:, :, :32, :, :]
#         slow_pathway = fast_pathway[:, :, ::4, :, :]
#         return [slow_pathway, fast_pathway]
#     elif model_name == 'r3d_18':
#         return videos[:, :, ::4, :, :]
#     elif model_name == 'I3D':
#         return videos[:, :, ::2, :, :]
#     else:
#         return videos


# def plot_confusion_matrix(cm, classes, save_path):
#     """繪製混淆矩陣"""
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=classes, yticklabels=classes)
#     plt.title('Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()


# # ============================================================================
# # 訓練函數（加入 AMP）
# # ============================================================================
# def train_model(model, train_loader, val_loader, device, model_name, num_epochs, patience, classes, save_dir, use_amp=True):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
#     # 使用 OneCycleLR 學習率調度器
#     scheduler = optim.lr_scheduler.OneCycleLR(
#         optimizer, 
#         max_lr=1e-3, 
#         epochs=num_epochs, 
#         steps_per_epoch=len(train_loader),
#         pct_start=0.3
#     )
    
#     # 初始化 AMP Scaler
#     scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
#     history = defaultdict(list)
#     best_val_loss = float('inf')
#     best_val_acc = 0.0
#     patience_counter = 0
    
#     print(f"\n{'='*70}")
#     print(f"  開始訓練 {model_name}")
#     print(f"  混合精度訓練: {'啟用' if use_amp else '關閉'}")
#     print(f"{'='*70}")
    
#     for epoch in range(num_epochs):
#         # -------------------- 訓練階段 --------------------
#         model.train()
#         train_loss = 0.0
#         train_top1 = 0.0
#         train_top5 = 0.0
#         train_samples = 0
#         train_probs = []
#         train_labels = []
#         train_preds = []
        
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
#         for videos, labels in pbar:
#             if videos is None or labels is None:
#                 continue
#             videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)
#             optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

#             # 使用 AMP
#             with torch.cuda.amp.autocast(enabled=use_amp):
#                 inputs = prepare_inputs(model_name, videos)
#                 outputs = model(inputs)
                
#                 if model_name == 'I3D':
#                     outputs = outputs.mean(dim=2)

#                 loss = criterion(outputs, labels)

#             # AMP backward
#             scaler.scale(loss).backward()
            
#             # 梯度裁剪
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             scaler.step(optimizer)
#             scaler.update()
#             scheduler.step()
            
#             train_loss += loss.item() * videos.size(0)
#             train_top1 += top_k_accuracy(outputs, labels, k=1) * videos.size(0)
#             train_top5 += top_k_accuracy(outputs, labels, k=5) * videos.size(0)
#             train_samples += videos.size(0)

#             probs = torch.softmax(outputs, dim=1).cpu().detach().numpy()
#             train_probs.extend(probs)
#             train_labels.extend(labels.cpu().numpy())
#             _, preds = torch.max(outputs, 1)
#             train_preds.extend(preds.cpu().numpy())
            
#             # 更新進度條
#             pbar.set_postfix({
#                 'loss': f'{loss.item():.4f}',
#                 'acc': f'{top_k_accuracy(outputs, labels, k=1):.2f}%'
#             })
        
#         train_loss = train_loss / train_samples if train_samples > 0 else float('inf')
#         train_top1 = train_top1 / train_samples if train_samples > 0 else 0.0
#         train_top5 = train_top5 / train_samples if train_samples > 0 else 0.0
        
#         # 計算訓練AUC
#         unique_labels = sorted(set(train_labels))
#         filtered_classes = [classes[i] for i in unique_labels]
#         train_metrics = {}
#         train_auc = {}
        
#         if unique_labels:
#             report = classification_report(train_labels, train_preds, 
#                                           labels=unique_labels, 
#                                           target_names=filtered_classes, 
#                                           output_dict=True, zero_division=0)
#             train_precision = {f'precision_class_{class_name}': report[class_name]['precision'] 
#                              for class_name in filtered_classes}
            
#             try:
#                 for idx, class_name in enumerate(filtered_classes):
#                     i = unique_labels[idx]
#                     if sum(1 for x in train_labels if x == i) > 0:
#                         auc_score = roc_auc_score(
#                             [1 if x == i else 0 for x in train_labels],
#                             [p[i] for p in train_probs]
#                         )
#                         train_auc[f'auc_class_{class_name}'] = auc_score
                
#                 for class_name in filtered_classes:
#                     train_metrics[class_name] = report[class_name]
#                     train_metrics[class_name]['auc'] = train_auc.get(f'auc_class_{class_name}', 0.0)
#             except ValueError as e:
#                 print(f"Warning: Train AUC calculation failed: {e}")
#                 train_auc = {f'auc_class_{class_name}': 0.0 for class_name in filtered_classes}

#         # -------------------- 驗證階段 --------------------
#         model.eval()
#         val_loss = 0.0
#         val_top1 = 0.0
#         val_top5 = 0.0
#         val_samples = 0
#         val_preds = []
#         val_labels = []
#         val_probs = []
        
#         with torch.no_grad():
#             for videos, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
#                 if videos is None or labels is None:
#                     continue
#                 videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
#                 with torch.cuda.amp.autocast(enabled=use_amp):
#                     inputs = prepare_inputs(model_name, videos)
#                     outputs = model(inputs)
                    
#                     if model_name == 'I3D':
#                         outputs = outputs.mean(dim=2)

#                     loss = criterion(outputs, labels)
                
#                 val_loss += loss.item() * videos.size(0)
#                 val_top1 += top_k_accuracy(outputs, labels, k=1) * videos.size(0)
#                 val_top5 += top_k_accuracy(outputs, labels, k=5) * videos.size(0)
#                 val_samples += videos.size(0)

#                 probs = torch.softmax(outputs, dim=1).cpu().detach().numpy()
#                 val_probs.extend(probs)
#                 val_labels.extend(labels.cpu().numpy())
#                 _, preds = torch.max(outputs, 1)
#                 val_preds.extend(preds.cpu().numpy())
        
#         val_loss = val_loss / val_samples if val_samples > 0 else float('inf')
#         val_top1 = val_top1 / val_samples if val_samples > 0 else 0.0
#         val_top5 = val_top5 / val_samples if val_samples > 0 else 0.0

#         # 計算驗證AUC
#         unique_labels = sorted(set(val_labels))
#         filtered_classes = [classes[i] for i in unique_labels]
#         val_metrics = {}
#         val_auc = {}
        
#         if unique_labels:
#             report = classification_report(val_labels, val_preds, 
#                                           labels=unique_labels, 
#                                           target_names=filtered_classes, 
#                                           output_dict=True, zero_division=0)
#             val_precision = {f'precision_{class_name}': report[class_name]['precision'] 
#                            for class_name in filtered_classes}
            
#             try:
#                 for idx, class_name in enumerate(filtered_classes):
#                     i = unique_labels[idx]
#                     if sum(1 for x in val_labels if x == i) > 0:
#                         auc_score = roc_auc_score(
#                             [1 if x == i else 0 for x in val_labels],
#                             [p[i] for p in val_probs]
#                         )
#                         val_auc[f'auc_{class_name}'] = auc_score
                
#                 for class_name in filtered_classes:
#                     val_metrics[class_name] = report[class_name]
#                     val_metrics[class_name]['auc'] = val_auc.get(f'auc_{class_name}', 0.0)
#             except ValueError as e:
#                 print(f"Warning: Val AUC calculation failed: {e}")
#                 val_auc = {f'auc_{class_name}': 0.0 for class_name in filtered_classes}
#                 for class_name in filtered_classes:
#                     val_metrics[class_name] = report[class_name]
#                     val_metrics[class_name]['auc'] = 0.0
        
#         # -------------------- 記錄結果 --------------------
#         history['epoch'].append(epoch + 1)
#         history['train_loss'].append(train_loss)
#         history['train_top1'].append(train_top1)
#         history['train_top5'].append(train_top5)
#         history['val_loss'].append(val_loss)
#         history['val_top1'].append(val_top1)
#         history['val_top5'].append(val_top5)
#         history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
#         for key, value in train_precision.items():
#             history[f'train_{key}'].append(value)
#         for key, value in val_precision.items():
#             history[f'val_{key}'].append(value)
#         for key, value in train_auc.items():
#             history[f'train_{key}'].append(value)
#         for key, value in val_auc.items():
#             history[f'val_{key}'].append(value)
        
#         # -------------------- 輸出進度 --------------------
#         print(f"\nEpoch {epoch+1}/{num_epochs}")
#         print(f"  Train Loss: {train_loss:.4f} | Acc: {train_top1:.2f}% | Top-5: {train_top5:.2f}%")
#         print(f"  Val Loss:   {val_loss:.4f} | Acc: {val_top1:.2f}% | Top-5: {val_top5:.2f}%")
#         print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
#         print("\n  Train Metrics (per class):")
#         print(f"  {'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
#         print("  " + "-" * 55)
#         for class_name in filtered_classes[:3]:
#             metrics = train_metrics.get(class_name, {})
#             print(f"  {class_name:<15} {metrics.get('precision', 0.0):<10.4f} "
#                   f"{metrics.get('recall', 0.0):<10.4f} {metrics.get('f1-score', 0.0):<10.4f} "
#                   f"{metrics.get('auc', 0.0):<10.4f}")
#         print("  ... (更多詳情請見CSV)")
        
#         # -------------------- 早停機制 --------------------
#         if val_top1 > best_val_acc:
#             best_val_acc = val_top1
#             best_val_loss = val_loss
#             patience_counter = 0
            
#             model_path = os.path.join(save_dir, f'best_model_{model_name}.pth')
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_acc': val_top1,
#                 'val_loss': val_loss,
#                 'train_acc': train_top1,
#             }, model_path)
#             print(f"  ✅ 保存最佳模型 (Val Acc: {val_top1:.2f}%)")
#         else:
#             patience_counter += 1
#             print(f"  ⏳ Patience: {patience_counter}/{patience}")
#             if patience_counter >= patience:
#                 print(f"\n  ⏹️ Early stopping at epoch {epoch+1}")
#                 print(f"  🏆 Best Val Acc: {best_val_acc:.2f}%")
#                 break
    
#     # 保存訓練歷史
#     history_path = os.path.join(save_dir, f'training_history_{model_name}.csv')
#     pd.DataFrame(history).to_csv(history_path, index=False)
#     print(f"\n✅ 訓練完成！")
#     print(f"   最佳驗證準確率: {best_val_acc:.2f}%")
#     print(f"   訓練歷史已保存: {history_path}")
#     return history


# # ============================================================================
# # 測試函數（加入混淆矩陣）
# # ============================================================================
# def test_model(model, test_loader, device, classes, model_name, save_dir, use_amp=True):
#     print(f"\n{'='*70}")
#     print(f"  測試 {model_name}")
#     print(f"{'='*70}")
    
#     model.eval()
#     test_loss = 0.0
#     test_top1 = 0.0
#     test_top5 = 0.0
#     criterion = nn.CrossEntropyLoss()
#     all_preds = []
#     all_labels = []
#     all_probs = []
#     test_samples = 0
    
#     with torch.no_grad():
#         for videos, labels in tqdm(test_loader, desc="Testing"):
#             if videos is None or labels is None:
#                 continue
#             videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
#             with torch.cuda.amp.autocast(enabled=use_amp):
#                 inputs = prepare_inputs(model_name, videos)
#                 outputs = model(inputs)
                
#                 if model_name == 'I3D':
#                     outputs = outputs.mean(dim=2)

#                 loss = criterion(outputs, labels)
            
#             test_loss += loss.item() * videos.size(0)
#             test_top1 += top_k_accuracy(outputs, labels, k=1) * videos.size(0)
#             test_top5 += top_k_accuracy(outputs, labels, k=5) * videos.size(0)
#             test_samples += videos.size(0)
            
#             probs = torch.softmax(outputs, dim=1).cpu().numpy()
#             all_probs.extend(probs)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
    
#     test_loss = test_loss / test_samples
#     test_top1 = test_top1 / test_samples
#     test_top5 = test_top5 / test_samples
    
#     # 計算混淆矩陣
#     cm = confusion_matrix(all_labels, all_preds)
#     cm_path = os.path.join(save_dir, f'confusion_matrix_{model_name}.png')
#     plot_confusion_matrix(cm, classes, cm_path)
    
#     # 計算測試AUC
#     unique_labels = sorted(set(all_labels))
#     filtered_classes = [classes[i] for i in unique_labels]
#     report = classification_report(all_labels, all_preds, 
#                                    target_names=filtered_classes,
#                                    digits=4,
#                                    labels=unique_labels,
#                                    output_dict=True, zero_division=0)
    
#     test_auc = {}
#     test_metrics = {}
#     try:
#         for idx, class_name in enumerate(filtered_classes):
#             i = unique_labels[idx]
#             if sum(1 for x in all_labels if x == i) > 0:
#                 auc_score = roc_auc_score(
#                     [1 if x == i else 0 for x in all_labels],
#                     [p[i] for p in all_probs]
#                 )
#                 test_auc[class_name] = auc_score
        
#         for class_name in filtered_classes:
#             test_metrics[class_name] = report[class_name]
#             test_metrics[class_name]['auc'] = test_auc.get(class_name, 0.0)
#     except ValueError as e:
#         print(f"Warning: Test AUC calculation failed: {e}")
#         for class_name in filtered_classes:
#             test_metrics[class_name] = report[class_name]
#             test_metrics[class_name]['auc'] = 0.0
    
#     # 輸出結果
#     print(f"\n📊 測試結果")
#     print(f"  Test Loss: {test_loss:.4f}")
#     print(f"  Test Accuracy: {test_top1:.2f}%")
#     print(f"  Test Top-5 Accuracy: {test_top5:.2f}%")
#     print(f"\n  各類別性能:")
#     print(f"  {'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}")
#     print("  " + "-" * 55)
#     for class_name in filtered_classes:
#         metrics = test_metrics[class_name]
#         print(f"  {class_name:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
#               f"{metrics['f1-score']:<10.4f} {metrics.get('auc', 0.0):<10.4f}")
    
#     # 保存測試報告
#     report_path = os.path.join(save_dir, f'test_report_{model_name}.txt')
#     with open(report_path, 'w', encoding='utf-8') as f:
#         f.write(f"Test Loss: {test_loss:.4f}\n")
#         f.write(f"Test Accuracy: {test_top1:.2f}%\n")
#         f.write(f"Test Top-5 Accuracy: {test_top5:.2f}%\n\n")
#         f.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}\n")
#         f.write("-" * 60 + "\n")
#         for class_name in filtered_classes:
#             metrics = test_metrics[class_name]
#             f.write(f"{class_name:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
#                    f"{metrics['f1-score']:<10.4f} {metrics.get('auc', 0.0):<10.4f}\n")
    
#     print(f"\n  ✅ 測試報告已保存: {report_path}")
#     print(f"  ✅ 混淆矩陣已保存: {cm_path}")
#     return test_top1


# # ============================================================================
# # 主函數
# # ============================================================================
# def main():
#     # 路徑設定
#     root_dir = r'D:\GAN\video\rembg'
#     save_dir = r'D:\GAN\video\results1022'
    
#     # 自動創建結果資料夾
#     os.makedirs(save_dir, exist_ok=True)
    
#     # 超參數
#     clip_len = 64
#     batch_size = 8
#     num_epochs = 100
#     patience = 10
#     use_amp = True  # 啟用自動混合精度訓練
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     print(f"\n{'#'*70}")
#     print(f"# 工業動作識別訓練系統（優化版）")
#     print(f"# 基於論文 + PyTorch 最佳實踐")
#     print(f"{'#'*70}")
#     print(f"\n🖥️ 設備資訊:")
#     print(f"  使用設備: {device}")
#     if torch.cuda.is_available():
#         print(f"  GPU: {torch.cuda.get_device_name(0)}")
#         print(f"  CUDA記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
#         print(f"  cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    
#     print(f"\n⚙️ 訓練參數:")
#     print(f"  Clip Length: {clip_len} frames")
#     print(f"  Batch Size: {batch_size}")
#     print(f"  Max Epochs: {num_epochs}")
#     print(f"  Patience: {patience}")
#     print(f"  Learning Rate: 1e-4 (AdamW)")
#     print(f"  混合精度訓練: {'啟用' if use_amp else '關閉'}")
#     print(f"  梯度裁剪: 啟用 (max_norm=1.0)")
#     print(f"  學習率調度: OneCycleLR")
#     print(f"  結果保存位置: {save_dir}")
    
#     # 數據增強
#     train_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         # transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomRotation(degrees=10),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     val_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # 載入數據集
#     print(f"\n📂 載入數據集...")
#     train_dataset = VideoDataset(root_dir, split='train', clip_len=clip_len, transform=train_transform)
#     val_dataset = VideoDataset(root_dir, split='val', clip_len=clip_len, transform=val_transform)
#     test_dataset = VideoDataset(root_dir, split='test', clip_len=clip_len, transform=val_transform)

#     if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
#         print("❌ 找不到影片，請檢查數據集路徑")
#         return

#     # 根據系統調整 num_workers
#     if platform.system() == "Windows":
#         num_workers = min(4, os.cpu_count() // 2)  # Windows 建議較少 workers
#     else:
#         num_workers = min(8, os.cpu_count())
    
#     print(f"\n  使用 {num_workers} 個 DataLoader workers")
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
#                              num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
#                            num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
#                             num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False)

#     print(f"\n📊 數據集統計:")
#     print(f"  Classes: {train_dataset.classes}")
#     print(f"  Train: {len(train_dataset)} videos")
#     print(f"  Val:   {len(val_dataset)} videos")
#     print(f"  Test:  {len(test_dataset)} videos")

#     # 模型字典
#     models_dict = {
#         'I3D': InceptionI3d(num_classes=400, in_channels=3),
#         'r3d_18': models.video.r3d_18(weights=R3D_18_Weights.KINETICS400_V1),
#         'slowfast_r50': slowfast_r50(pretrained=True)
#     }

#     # 訓練3個模型
#     results_summary = []
    
#     for model_name in ['I3D', 'r3d_18', 'slowfast_r50']:
#         print(f"\n{'#'*70}")
#         print(f"# 模型: {model_name}")
#         print(f"{'#'*70}")
        
#         model = models_dict[model_name]
        
#         # 載入I3D預訓練權重
#         if model_name == 'I3D':
#             try:
#                 pretrained_dict = torch.load('models/rgb_imagenet.pt')
#                 model.load_state_dict(pretrained_dict)
#                 print("✅ 載入I3D預訓練權重")
#             except FileNotFoundError:
#                 print("⚠️ 找不到預訓練權重，使用隨機初始化")
        
#         model = model.to(device)
        
#         # 修改分類層
#         num_classes = len(train_dataset.classes)
#         if model_name == 'r3d_18':
#             model.fc = nn.Sequential(
#                 nn.Dropout(0.5),
#                 nn.Linear(model.fc.in_features, num_classes))
#         elif model_name == 'I3D':
#             model.logits = nn.Sequential(
#                 nn.Dropout(0.5),
#                 nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True))
#         elif model_name == 'slowfast_r50':
#             model.blocks[-1].proj = nn.Sequential(
#                 nn.Dropout(0.5),
#                 nn.Linear(model.blocks[-1].proj.in_features, num_classes))

#         model = model.to(device)
        
#         # 訓練模型
#         start_time = time.time()
#         history = train_model(model, train_loader, val_loader, device, model_name, 
#                              num_epochs, patience, train_dataset.classes, save_dir, use_amp=use_amp)
#         training_time = time.time() - start_time
        
#         # 載入最佳模型並測試
#         model_path = os.path.join(save_dir, f'best_model_{model_name}.pth')
#         checkpoint = torch.load(model_path)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         test_acc = test_model(model, test_loader, device, train_dataset.classes, model_name, save_dir, use_amp=use_amp)
        
#         # 記錄結果摘要
#         results_summary.append({
#             'Model': model_name,
#             'Best Val Acc': checkpoint['val_acc'],
#             'Test Acc': test_acc,
#             'Training Time (hrs)': training_time / 3600,
#             'Epochs Trained': checkpoint['epoch'] + 1
#         })
        
#         print(f"\n✅ {model_name} 完成！")
#         print(f"   最佳驗證準確率: {checkpoint['val_acc']:.2f}%")
#         print(f"   測試準確率: {test_acc:.2f}%")
#         print(f"   訓練時間: {training_time/3600:.2f} 小時")
        
#         # 清理GPU記憶體
#         del model
#         torch.cuda.empty_cache()
#         gc.collect()
    
#     # 保存總結報告
#     summary_df = pd.DataFrame(results_summary)
#     summary_path = os.path.join(save_dir, 'results_summary.csv')
#     summary_df.to_csv(summary_path, index=False)
    
#     print(f"\n{'='*70}")
#     print(f"🎉 所有模型訓練完成！")
#     print(f"{'='*70}")
#     print(f"\n📊 結果摘要:")
#     print(summary_df.to_string(index=False))
#     print(f"\n💾 所有結果已保存至: {save_dir}")
#     print(f"   - 模型權重: best_model_*.pth")
#     print(f"   - 訓練曲線: training_history_*.csv")
#     print(f"   - 測試報告: test_report_*.txt")
#     print(f"   - 混淆矩陣: confusion_matrix_*.png")
#     print(f"   - 總結報告: results_summary.csv")


# if __name__ == "__main__":
#     main()
    # Data transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    train_dataset = VideoDataset(data_root, split='train', clip_len=clip_length, transforms=transform)
    val_dataset = VideoDataset(data_root, split='val', clip_len=clip_length, transforms=transform)
    test_dataset = VideoDataset(data_root, split='test', clip_len=clip_length, transforms=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    # Create model
    num_classes = len(train_dataset.classes)
    print(f'Number of classes: {num_classes}')
    model = get_model(num_classes=num_classes, pretrained=True)
    
    # Freeze most layers, only train last few
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'layer4' or 'fc' in name:  #如果
            param.requires_grad = True 
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train model
    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        test_loader=test_loader,  # 傳入 test_loader
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        patience=early_stopping_patience,
        device=device
    )
    
    # Evaluate and visualize results
    print("\nEvaluating model on test dataset...")
    acc1, acc5, test_report = evaluate_model(model, test_loader, device, run_id)
    
    print(f"\nACC1 (Top-1 Accuracy): {acc1:.4f} ({acc1*100:.2f}%)")
    print(f"ACC5 (Top-5 Accuracy): {acc5:.4f} ({acc5*100:.2f}%)")
    print("\nDetailed Classification Report:")
    print(test_report)
    
    # Save results and model
    with open(f'test_results_{run_id}.txt', 'w') as f:
        f.write(f"ACC1 (Top-1 Accuracy): {acc1:.4f} ({acc1*100:.2f}%)\n")
        f.write(f"ACC5 (Top-5 Accuracy): {acc5:.4f} ({acc5*100:.2f}%)\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(test_report)
    
    plot_training_history(history, run_id)
    torch.save(model.state_dict(), f'{run_id}.pth')
    
    # Total execution time
    # total_time = time.time() - total_start_time
    # hours, remainder = divmod(total_time, 3600)
    # minutes, seconds = divmod(remainder, 60)
    # print(f'Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s')
    print('Model saved!')
    print(f'{run_id} is done~~~~~~~~~~~~~~~~~~~~~')

if __name__ == '__main__':
    main()








# import os
# import gc
# import time
# import torch
# import platform
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import torch.nn as nn
# import torch.optim as optim
# from collections import defaultdict
# from torchvision.io import read_video
# from torch.utils.data import DataLoader
# from torchvision import models, transforms
# from pytorchvideo.models.hub import slowfast_r50
# from torchvision.models.video import R3D_18_Weights
# from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# import sys
# sys.path.append('D:/GAN/pytorch-i3d')
# from pytorch_i3d import InceptionI3d

# # 啟用 cudnn benchmark 加速
# torch.backends.cudnn.benchmark = True

# # ============================================================================
# # VideoDataset類別（支援MP4）
# # ============================================================================
# class VideoDataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir, split='train', clip_len=64, transform=None):
#         """
#         支援MP4影片的資料集
#         資料夾結構: root_dir/split/class_name/video.mp4
#         """
#         self.root_dir = os.path.join(root_dir, split)
#         self.clip_len = clip_len
#         self.transform = transform
#         self.classes = sorted(os.listdir(self.root_dir))
#         self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
#         self.videos = []
        
#         for cls_name in self.classes:
#             cls_dir = os.path.join(self.root_dir, cls_name)
#             if not os.path.isdir(cls_dir):
#                 continue
#             for video_name in os.listdir(cls_dir):
#                 if video_name.endswith('.mp4'):
#                     self.videos.append((os.path.join(cls_dir, video_name), cls_name))
        
#         print(f"  {split.upper()}: {len(self.videos)} videos")

#     def __len__(self):
#         return len(self.videos)

#     def __getitem__(self, idx):
#         video_path, cls_name = self.videos[idx]
#         label = self.class_to_idx[cls_name]

#         try:
#             video, _, _ = read_video(video_path, pts_unit='sec')
#             total_frames = video.shape[0]
#         except Exception as e:
#             print(f"Error reading {video_path}: {e}")
#             return None, None

#         if total_frames == 0:
#             print(f"Warning: {video_path} has 0 frames")
#             return None, None

#         # Clip長度處理
#         if total_frames < self.clip_len:
#             pad_len = self.clip_len - total_frames
#             last_frame = video[-1:].repeat(pad_len, 1, 1, 1)
#             video = torch.cat([video, last_frame], dim=0)
#         else:
#             start_idx = np.random.randint(0, total_frames - self.clip_len + 1)
#             video = video[start_idx:start_idx + self.clip_len]

#         # (T, H, W, C) → (C, T, H, W)
#         video = video.permute(3, 0, 1, 2).float() / 255.0

#         if self.transform:
#             video = torch.stack([self.transform(video[:, i]) for i in range(video.shape[1])], dim=1)

#         return video, label


# # ============================================================================
# # 輔助函數
# # ============================================================================
# def top_k_accuracy(output, target, k=1):
#     with torch.no_grad():
#         _, pred = output.topk(k, dim=1, largest=True, sorted=True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#         correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#         return correct_k.mul_(100.0 / output.size(0)).item()


# def prepare_inputs(model_name, videos):
#     """根據論文附錄準備輸入"""
#     if model_name == 'slowfast_r50':
#         fast_pathway = videos[:, :, :32, :, :]
#         slow_pathway = fast_pathway[:, :, ::4, :, :]
#         return [slow_pathway, fast_pathway]
#     elif model_name == 'r3d_18':
#         return videos[:, :, ::4, :, :]
#     elif model_name == 'I3D':
#         return videos[:, :, ::2, :, :]
#     else:
#         return videos


# def plot_confusion_matrix(cm, classes, save_path):
#     """繪製混淆矩陣"""
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=classes, yticklabels=classes)
#     plt.title('Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()


# # ============================================================================
# # 訓練函數（加入 AMP）
# # ============================================================================
# def train_model(model, train_loader, val_loader, device, model_name, num_epochs, patience, classes, save_dir, use_amp=True):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
#     # 使用 OneCycleLR 學習率調度器
#     scheduler = optim.lr_scheduler.OneCycleLR(
#         optimizer, 
#         max_lr=1e-3, 
#         epochs=num_epochs, 
#         steps_per_epoch=len(train_loader),
#         pct_start=0.3
#     )
    
#     # 初始化 AMP Scaler
#     scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
#     history = defaultdict(list)
#     best_val_loss = float('inf')
#     best_val_acc = 0.0
#     patience_counter = 0
    
#     print(f"\n{'='*70}")
#     print(f"  開始訓練 {model_name}")
#     print(f"  混合精度訓練: {'啟用' if use_amp else '關閉'}")
#     print(f"{'='*70}")
    
#     for epoch in range(num_epochs):
#         # -------------------- 訓練階段 --------------------
#         model.train()
#         train_loss = 0.0
#         train_top1 = 0.0
#         train_top5 = 0.0
#         train_samples = 0
#         train_probs = []
#         train_labels = []
#         train_preds = []
        
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
#         for videos, labels in pbar:
#             if videos is None or labels is None:
#                 continue
#             videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)
#             optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

#             # 使用 AMP
#             with torch.cuda.amp.autocast(enabled=use_amp):
#                 inputs = prepare_inputs(model_name, videos)
#                 outputs = model(inputs)
                
#                 if model_name == 'I3D':
#                     outputs = outputs.mean(dim=2)

#                 loss = criterion(outputs, labels)

#             # AMP backward
#             scaler.scale(loss).backward()
            
#             # 梯度裁剪
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             scaler.step(optimizer)
#             scaler.update()
#             scheduler.step()
            
#             train_loss += loss.item() * videos.size(0)
#             train_top1 += top_k_accuracy(outputs, labels, k=1) * videos.size(0)
#             train_top5 += top_k_accuracy(outputs, labels, k=5) * videos.size(0)
#             train_samples += videos.size(0)

#             probs = torch.softmax(outputs, dim=1).cpu().detach().numpy()
#             train_probs.extend(probs)
#             train_labels.extend(labels.cpu().numpy())
#             _, preds = torch.max(outputs, 1)
#             train_preds.extend(preds.cpu().numpy())
            
#             # 更新進度條
#             pbar.set_postfix({
#                 'loss': f'{loss.item():.4f}',
#                 'acc': f'{top_k_accuracy(outputs, labels, k=1):.2f}%'
#             })
        
#         train_loss = train_loss / train_samples if train_samples > 0 else float('inf')
#         train_top1 = train_top1 / train_samples if train_samples > 0 else 0.0
#         train_top5 = train_top5 / train_samples if train_samples > 0 else 0.0
        
#         # 計算訓練AUC
#         unique_labels = sorted(set(train_labels))
#         filtered_classes = [classes[i] for i in unique_labels]
#         train_metrics = {}
#         train_auc = {}
        
#         if unique_labels:
#             report = classification_report(train_labels, train_preds, 
#                                           labels=unique_labels, 
#                                           target_names=filtered_classes, 
#                                           output_dict=True, zero_division=0)
#             train_precision = {f'precision_class_{class_name}': report[class_name]['precision'] 
#                              for class_name in filtered_classes}
            
#             try:
#                 for idx, class_name in enumerate(filtered_classes):
#                     i = unique_labels[idx]
#                     if sum(1 for x in train_labels if x == i) > 0:
#                         auc_score = roc_auc_score(
#                             [1 if x == i else 0 for x in train_labels],
#                             [p[i] for p in train_probs]
#                         )
#                         train_auc[f'auc_class_{class_name}'] = auc_score
                
#                 for class_name in filtered_classes:
#                     train_metrics[class_name] = report[class_name]
#                     train_metrics[class_name]['auc'] = train_auc.get(f'auc_class_{class_name}', 0.0)
#             except ValueError as e:
#                 print(f"Warning: Train AUC calculation failed: {e}")
#                 train_auc = {f'auc_class_{class_name}': 0.0 for class_name in filtered_classes}

#         # -------------------- 驗證階段 --------------------
#         model.eval()
#         val_loss = 0.0
#         val_top1 = 0.0
#         val_top5 = 0.0
#         val_samples = 0
#         val_preds = []
#         val_labels = []
#         val_probs = []
        
#         with torch.no_grad():
#             for videos, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
#                 if videos is None or labels is None:
#                     continue
#                 videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
#                 with torch.cuda.amp.autocast(enabled=use_amp):
#                     inputs = prepare_inputs(model_name, videos)
#                     outputs = model(inputs)
                    
#                     if model_name == 'I3D':
#                         outputs = outputs.mean(dim=2)

#                     loss = criterion(outputs, labels)
                
#                 val_loss += loss.item() * videos.size(0)
#                 val_top1 += top_k_accuracy(outputs, labels, k=1) * videos.size(0)
#                 val_top5 += top_k_accuracy(outputs, labels, k=5) * videos.size(0)
#                 val_samples += videos.size(0)

#                 probs = torch.softmax(outputs, dim=1).cpu().detach().numpy()
#                 val_probs.extend(probs)
#                 val_labels.extend(labels.cpu().numpy())
#                 _, preds = torch.max(outputs, 1)
#                 val_preds.extend(preds.cpu().numpy())
        
#         val_loss = val_loss / val_samples if val_samples > 0 else float('inf')
#         val_top1 = val_top1 / val_samples if val_samples > 0 else 0.0
#         val_top5 = val_top5 / val_samples if val_samples > 0 else 0.0

#         # 計算驗證AUC
#         unique_labels = sorted(set(val_labels))
#         filtered_classes = [classes[i] for i in unique_labels]
#         val_metrics = {}
#         val_auc = {}
        
#         if unique_labels:
#             report = classification_report(val_labels, val_preds, 
#                                           labels=unique_labels, 
#                                           target_names=filtered_classes, 
#                                           output_dict=True, zero_division=0)
#             val_precision = {f'precision_{class_name}': report[class_name]['precision'] 
#                            for class_name in filtered_classes}
            
#             try:
#                 for idx, class_name in enumerate(filtered_classes):
#                     i = unique_labels[idx]
#                     if sum(1 for x in val_labels if x == i) > 0:
#                         auc_score = roc_auc_score(
#                             [1 if x == i else 0 for x in val_labels],
#                             [p[i] for p in val_probs]
#                         )
#                         val_auc[f'auc_{class_name}'] = auc_score
                
#                 for class_name in filtered_classes:
#                     val_metrics[class_name] = report[class_name]
#                     val_metrics[class_name]['auc'] = val_auc.get(f'auc_{class_name}', 0.0)
#             except ValueError as e:
#                 print(f"Warning: Val AUC calculation failed: {e}")
#                 val_auc = {f'auc_{class_name}': 0.0 for class_name in filtered_classes}
#                 for class_name in filtered_classes:
#                     val_metrics[class_name] = report[class_name]
#                     val_metrics[class_name]['auc'] = 0.0
        
#         # -------------------- 記錄結果 --------------------
#         history['epoch'].append(epoch + 1)
#         history['train_loss'].append(train_loss)
#         history['train_top1'].append(train_top1)
#         history['train_top5'].append(train_top5)
#         history['val_loss'].append(val_loss)
#         history['val_top1'].append(val_top1)
#         history['val_top5'].append(val_top5)
#         history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
#         for key, value in train_precision.items():
#             history[f'train_{key}'].append(value)
#         for key, value in val_precision.items():
#             history[f'val_{key}'].append(value)
#         for key, value in train_auc.items():
#             history[f'train_{key}'].append(value)
#         for key, value in val_auc.items():
#             history[f'val_{key}'].append(value)
        
#         # -------------------- 輸出進度 --------------------
#         print(f"\nEpoch {epoch+1}/{num_epochs}")
#         print(f"  Train Loss: {train_loss:.4f} | Acc: {train_top1:.2f}% | Top-5: {train_top5:.2f}%")
#         print(f"  Val Loss:   {val_loss:.4f} | Acc: {val_top1:.2f}% | Top-5: {val_top5:.2f}%")
#         print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
#         print("\n  Train Metrics (per class):")
#         print(f"  {'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
#         print("  " + "-" * 55)
#         for class_name in filtered_classes[:3]:
#             metrics = train_metrics.get(class_name, {})
#             print(f"  {class_name:<15} {metrics.get('precision', 0.0):<10.4f} "
#                   f"{metrics.get('recall', 0.0):<10.4f} {metrics.get('f1-score', 0.0):<10.4f} "
#                   f"{metrics.get('auc', 0.0):<10.4f}")
#         print("  ... (更多詳情請見CSV)")
        
#         # -------------------- 早停機制 --------------------
#         if val_top1 > best_val_acc:
#             best_val_acc = val_top1
#             best_val_loss = val_loss
#             patience_counter = 0
            
#             model_path = os.path.join(save_dir, f'best_model_{model_name}.pth')
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_acc': val_top1,
#                 'val_loss': val_loss,
#                 'train_acc': train_top1,
#             }, model_path)
#             print(f"  ✅ 保存最佳模型 (Val Acc: {val_top1:.2f}%)")
#         else:
#             patience_counter += 1
#             print(f"  ⏳ Patience: {patience_counter}/{patience}")
#             if patience_counter >= patience:
#                 print(f"\n  ⏹️ Early stopping at epoch {epoch+1}")
#                 print(f"  🏆 Best Val Acc: {best_val_acc:.2f}%")
#                 break
    
#     # 保存訓練歷史
#     history_path = os.path.join(save_dir, f'training_history_{model_name}.csv')
#     pd.DataFrame(history).to_csv(history_path, index=False)
#     print(f"\n✅ 訓練完成！")
#     print(f"   最佳驗證準確率: {best_val_acc:.2f}%")
#     print(f"   訓練歷史已保存: {history_path}")
#     return history


# # ============================================================================
# # 測試函數（加入混淆矩陣）
# # ============================================================================
# def test_model(model, test_loader, device, classes, model_name, save_dir, use_amp=True):
#     print(f"\n{'='*70}")
#     print(f"  測試 {model_name}")
#     print(f"{'='*70}")
    
#     model.eval()
#     test_loss = 0.0
#     test_top1 = 0.0
#     test_top5 = 0.0
#     criterion = nn.CrossEntropyLoss()
#     all_preds = []
#     all_labels = []
#     all_probs = []
#     test_samples = 0
    
#     with torch.no_grad():
#         for videos, labels in tqdm(test_loader, desc="Testing"):
#             if videos is None or labels is None:
#                 continue
#             videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
#             with torch.cuda.amp.autocast(enabled=use_amp):
#                 inputs = prepare_inputs(model_name, videos)
#                 outputs = model(inputs)
                
#                 if model_name == 'I3D':
#                     outputs = outputs.mean(dim=2)

#                 loss = criterion(outputs, labels)
            
#             test_loss += loss.item() * videos.size(0)
#             test_top1 += top_k_accuracy(outputs, labels, k=1) * videos.size(0)
#             test_top5 += top_k_accuracy(outputs, labels, k=5) * videos.size(0)
#             test_samples += videos.size(0)
            
#             probs = torch.softmax(outputs, dim=1).cpu().numpy()
#             all_probs.extend(probs)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
    
#     test_loss = test_loss / test_samples
#     test_top1 = test_top1 / test_samples
#     test_top5 = test_top5 / test_samples
    
#     # 計算混淆矩陣
#     cm = confusion_matrix(all_labels, all_preds)
#     cm_path = os.path.join(save_dir, f'confusion_matrix_{model_name}.png')
#     plot_confusion_matrix(cm, classes, cm_path)
    
#     # 計算測試AUC
#     unique_labels = sorted(set(all_labels))
#     filtered_classes = [classes[i] for i in unique_labels]
#     report = classification_report(all_labels, all_preds, 
#                                    target_names=filtered_classes,
#                                    digits=4,
#                                    labels=unique_labels,
#                                    output_dict=True, zero_division=0)
    
#     test_auc = {}
#     test_metrics = {}
#     try:
#         for idx, class_name in enumerate(filtered_classes):
#             i = unique_labels[idx]
#             if sum(1 for x in all_labels if x == i) > 0:
#                 auc_score = roc_auc_score(
#                     [1 if x == i else 0 for x in all_labels],
#                     [p[i] for p in all_probs]
#                 )
#                 test_auc[class_name] = auc_score
        
#         for class_name in filtered_classes:
#             test_metrics[class_name] = report[class_name]
#             test_metrics[class_name]['auc'] = test_auc.get(class_name, 0.0)
#     except ValueError as e:
#         print(f"Warning: Test AUC calculation failed: {e}")
#         for class_name in filtered_classes:
#             test_metrics[class_name] = report[class_name]
#             test_metrics[class_name]['auc'] = 0.0
    
#     # 輸出結果
#     print(f"\n📊 測試結果")
#     print(f"  Test Loss: {test_loss:.4f}")
#     print(f"  Test Accuracy: {test_top1:.2f}%")
#     print(f"  Test Top-5 Accuracy: {test_top5:.2f}%")
#     print(f"\n  各類別性能:")
#     print(f"  {'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}")
#     print("  " + "-" * 55)
#     for class_name in filtered_classes:
#         metrics = test_metrics[class_name]
#         print(f"  {class_name:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
#               f"{metrics['f1-score']:<10.4f} {metrics.get('auc', 0.0):<10.4f}")
    
#     # 保存測試報告
#     report_path = os.path.join(save_dir, f'test_report_{model_name}.txt')
#     with open(report_path, 'w', encoding='utf-8') as f:
#         f.write(f"Test Loss: {test_loss:.4f}\n")
#         f.write(f"Test Accuracy: {test_top1:.2f}%\n")
#         f.write(f"Test Top-5 Accuracy: {test_top5:.2f}%\n\n")
#         f.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}\n")
#         f.write("-" * 60 + "\n")
#         for class_name in filtered_classes:
#             metrics = test_metrics[class_name]
#             f.write(f"{class_name:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
#                    f"{metrics['f1-score']:<10.4f} {metrics.get('auc', 0.0):<10.4f}\n")
    
#     print(f"\n  ✅ 測試報告已保存: {report_path}")
#     print(f"  ✅ 混淆矩陣已保存: {cm_path}")
#     return test_top1


# # ============================================================================
# # 主函數
# # ============================================================================
# def main():
#     # 路徑設定
#     root_dir = r'D:\GAN\video\rembg'
#     save_dir = r'D:\GAN\video\results1022'
    
#     # 自動創建結果資料夾
#     os.makedirs(save_dir, exist_ok=True)
    
#     # 超參數
#     clip_len = 64
#     batch_size = 8
#     num_epochs = 100
#     patience = 10
#     use_amp = True  # 啟用自動混合精度訓練
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     print(f"\n{'#'*70}")
#     print(f"# 工業動作識別訓練系統（優化版）")
#     print(f"# 基於論文 + PyTorch 最佳實踐")
#     print(f"{'#'*70}")
#     print(f"\n🖥️ 設備資訊:")
#     print(f"  使用設備: {device}")
#     if torch.cuda.is_available():
#         print(f"  GPU: {torch.cuda.get_device_name(0)}")
#         print(f"  CUDA記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
#         print(f"  cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    
#     print(f"\n⚙️ 訓練參數:")
#     print(f"  Clip Length: {clip_len} frames")
#     print(f"  Batch Size: {batch_size}")
#     print(f"  Max Epochs: {num_epochs}")
#     print(f"  Patience: {patience}")
#     print(f"  Learning Rate: 1e-4 (AdamW)")
#     print(f"  混合精度訓練: {'啟用' if use_amp else '關閉'}")
#     print(f"  梯度裁剪: 啟用 (max_norm=1.0)")
#     print(f"  學習率調度: OneCycleLR")
#     print(f"  結果保存位置: {save_dir}")
    
#     # 數據增強
#     train_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         # transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomRotation(degrees=10),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     val_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # 載入數據集
#     print(f"\n📂 載入數據集...")
#     train_dataset = VideoDataset(root_dir, split='train', clip_len=clip_len, transform=train_transform)
#     val_dataset = VideoDataset(root_dir, split='val', clip_len=clip_len, transform=val_transform)
#     test_dataset = VideoDataset(root_dir, split='test', clip_len=clip_len, transform=val_transform)

#     if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
#         print("❌ 找不到影片，請檢查數據集路徑")
#         return

#     # 根據系統調整 num_workers
#     if platform.system() == "Windows":
#         num_workers = min(4, os.cpu_count() // 2)  # Windows 建議較少 workers
#     else:
#         num_workers = min(8, os.cpu_count())
    
#     print(f"\n  使用 {num_workers} 個 DataLoader workers")
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
#                              num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
#                            num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
#                             num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False)

#     print(f"\n📊 數據集統計:")
#     print(f"  Classes: {train_dataset.classes}")
#     print(f"  Train: {len(train_dataset)} videos")
#     print(f"  Val:   {len(val_dataset)} videos")
#     print(f"  Test:  {len(test_dataset)} videos")

#     # 模型字典
#     models_dict = {
#         'I3D': InceptionI3d(num_classes=400, in_channels=3),
#         'r3d_18': models.video.r3d_18(weights=R3D_18_Weights.KINETICS400_V1),
#         'slowfast_r50': slowfast_r50(pretrained=True)
#     }

#     # 訓練3個模型
#     results_summary = []
    
#     for model_name in ['I3D', 'r3d_18', 'slowfast_r50']:
#         print(f"\n{'#'*70}")
#         print(f"# 模型: {model_name}")
#         print(f"{'#'*70}")
        
#         model = models_dict[model_name]
        
#         # 載入I3D預訓練權重
#         if model_name == 'I3D':
#             try:
#                 pretrained_dict = torch.load('models/rgb_imagenet.pt')
#                 model.load_state_dict(pretrained_dict)
#                 print("✅ 載入I3D預訓練權重")
#             except FileNotFoundError:
#                 print("⚠️ 找不到預訓練權重，使用隨機初始化")
        
#         model = model.to(device)
        
#         # 修改分類層
#         num_classes = len(train_dataset.classes)
#         if model_name == 'r3d_18':
#             model.fc = nn.Sequential(
#                 nn.Dropout(0.5),
#                 nn.Linear(model.fc.in_features, num_classes))
#         elif model_name == 'I3D':
#             model.logits = nn.Sequential(
#                 nn.Dropout(0.5),
#                 nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True))
#         elif model_name == 'slowfast_r50':
#             model.blocks[-1].proj = nn.Sequential(
#                 nn.Dropout(0.5),
#                 nn.Linear(model.blocks[-1].proj.in_features, num_classes))

#         model = model.to(device)
        
#         # 訓練模型
#         start_time = time.time()
#         history = train_model(model, train_loader, val_loader, device, model_name, 
#                              num_epochs, patience, train_dataset.classes, save_dir, use_amp=use_amp)
#         training_time = time.time() - start_time
        
#         # 載入最佳模型並測試
#         model_path = os.path.join(save_dir, f'best_model_{model_name}.pth')
#         checkpoint = torch.load(model_path)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         test_acc = test_model(model, test_loader, device, train_dataset.classes, model_name, save_dir, use_amp=use_amp)
        
#         # 記錄結果摘要
#         results_summary.append({
#             'Model': model_name,
#             'Best Val Acc': checkpoint['val_acc'],
#             'Test Acc': test_acc,
#             'Training Time (hrs)': training_time / 3600,
#             'Epochs Trained': checkpoint['epoch'] + 1
#         })
        
#         print(f"\n✅ {model_name} 完成！")
#         print(f"   最佳驗證準確率: {checkpoint['val_acc']:.2f}%")
#         print(f"   測試準確率: {test_acc:.2f}%")
#         print(f"   訓練時間: {training_time/3600:.2f} 小時")
        
#         # 清理GPU記憶體
#         del model
#         torch.cuda.empty_cache()
#         gc.collect()
    
#     # 保存總結報告
#     summary_df = pd.DataFrame(results_summary)
#     summary_path = os.path.join(save_dir, 'results_summary.csv')
#     summary_df.to_csv(summary_path, index=False)
    
#     print(f"\n{'='*70}")
#     print(f"🎉 所有模型訓練完成！")
#     print(f"{'='*70}")
#     print(f"\n📊 結果摘要:")
#     print(summary_df.to_string(index=False))
#     print(f"\n💾 所有結果已保存至: {save_dir}")
#     print(f"   - 模型權重: best_model_*.pth")
#     print(f"   - 訓練曲線: training_history_*.csv")
#     print(f"   - 測試報告: test_report_*.txt")
#     print(f"   - 混淆矩陣: confusion_matrix_*.png")
#     print(f"   - 總結報告: results_summary.csv")


# if __name__ == "__main__":
#     main()