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
    """讀取視頻數據集"""
    def __init__(self, root_dir, split='train', clip_len=16, transforms=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transforms = transforms
        
        # 讀取資料夾結構
        folder_path = os.path.join(root_dir, split)
        self.classes = sorted(os.listdir(folder_path))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 掃描視頻檔案
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
            
            # 處理幀數不足的情況
            if video_frames.shape[0] < self.clip_len:
                video_frames = torch.cat([video_frames] * (self.clip_len // video_frames.shape[0] + 1), dim=0)
            
            # 隨機選擇視頻片段
            total_frames = video_frames.shape[0]
            start_idx = np.random.randint(0, total_frames - self.clip_len) if total_frames > self.clip_len else 0
            video_clip = video_frames[start_idx:start_idx + self.clip_len]
            
            # 確保長度為 clip_len
            if video_clip.shape[0] < self.clip_len:
                video_clip = torch.cat([video_clip, video_clip[-1].unsqueeze(0).repeat(self.clip_len - video_clip.shape[0], 1, 1, 1)], dim=0)
            elif video_clip.shape[0] > self.clip_len:
                video_clip = video_clip[:self.clip_len]
            
            # 應用轉換
            if self.transforms:
                transformed_frames = []
                for frame in video_clip:
                    frame = frame.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
                    frame = self.transforms(frame)
                    transformed_frames.append(frame)
                video_tensor = torch.stack(transformed_frames, dim=1)  # [C, T, H, W]
                return video_tensor, self.labels[index]
            
            # 無轉換情況
            video_clip = video_clip.permute(3, 0, 1, 2)  # [T, H, W, C] -> [C, T, H, W]
            return video_clip.float() / 255.0, self.labels[index]
        
        except Exception as e:
            print(f"Error loading video {self.fnames[index]}: {e}")
            return torch.zeros(3, self.clip_len, 224, 224), self.labels[index]


def get_model(num_classes, pretrained=True):
    """獲取3D ResNet模型"""
    try:
        from torchvision.models.video import r3d_18
        model = r3d_18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    except (ImportError, AttributeError):
        # 備選方案: 使用2D ResNet並轉換為3D
        from torchvision.models import resnet18
        model = resnet18(pretrained=pretrained)
        model.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model


def train_model(model, dataloaders, test_loader, criterion, optimizer, scheduler=None, num_epochs=25, patience=5, device='cuda'):
    """訓練模型"""
    model = model.to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'early stop {patience_counter}/{patience}')
        print('-' * 10) # 
        
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
                if scheduler:
                    scheduler.step()
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
                
    model.load_state_dict(best_model_wts)
    history['total_training_time'] = time.time() - start_time
    return model, history


def evaluate_model(model, test_loader, device='cuda', run_id=''):
    """評估模型性能"""
    model.eval()
    y_true, y_pred = [], []
    top5_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Top-1 accuracy
            _, preds = torch.max(outputs, 1)
            
            # Top-5 accuracy
            _, top5_preds = torch.topk(outputs, k=5, dim=1)
            batch_size = labels.size(0)
            for i in range(batch_size):
                if labels[i] in top5_preds[i]:
                    top5_correct += 1
            total_samples += batch_size
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # 計算指標
    acc1 = accuracy_score(y_true, y_pred)
    acc5 = top5_correct / total_samples
    report = classification_report(y_true, y_pred, target_names=test_loader.dataset.classes)
    
    # 繪製混淆矩陣
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
    
    # 繪製準確率對比
    plt.figure(figsize=(8, 6))
    accuracies = [acc1*100, acc5*100]
    labels_plot = ['ACC1 (Top-1)', 'ACC5 (Top-5)']
    bars = plt.bar(labels_plot, accuracies, color=['royalblue', 'forestgreen'], width=0.5)
    
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
    """繪製訓練歷史"""
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
    """主訓練函數"""
    # ============ 配置參數 ============
    run_id = 'run0115_finetune_EDA_byZiv' # 可以根據需要修改為不同的名稱
    data_root = r"/home/simslab/Desktop/api/Ziv/3D_ResNet/dataset"
    batch_size = 4
    num_epochs = 300
    early_stopping_patience = 10
    learning_rate = 0.0001
    clip_length = 16
    
    total_start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # ============ 1. 定義影像轉換 ============
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ============ 2. 創建數據加載器 ============
    train_dataset = VideoDataset(data_root, split='train', clip_len=clip_length, transforms=transform)
    val_dataset = VideoDataset(data_root, split='val', clip_len=clip_length, transforms=transform)
    test_dataset = VideoDataset(data_root, split='test', clip_len=clip_length, transforms=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    # ============ 3. 創建模型 ============
    num_classes = len(train_dataset.classes)
    print(f'Number of classes: {num_classes}')
    model = get_model(num_classes=num_classes, pretrained=True)
    
    # 凍結模型參數，只訓練最後幾層
    for param in model.parameters():
        param.requires_grad = False
    
    
    for name, param in model.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True
    
    # ============ 4. 定義損失函數、優化器和調度器 ============
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # ============ 5. 訓練模型 ============
    print("\nTraining started...")
    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        patience=early_stopping_patience,
        device=device
    )
    
    # ============ 6. 評估模型 ============
    print("\nEvaluating model on test dataset...")
    acc1, acc5, test_report = evaluate_model(model, test_loader, device, run_id)
    
    print(f"\nACC1 (Top-1 Accuracy): {acc1:.4f} ({acc1*100:.2f}%)")
    print(f"ACC5 (Top-5 Accuracy): {acc5:.4f} ({acc5*100:.2f}%)")
    print("\nDetailed Classification Report:")
    print(test_report)
    
    # ============ 7. 保存結果 ============
    with open(f'test_results_{run_id}.txt', 'w') as f:
        f.write(f"ACC1 (Top-1 Accuracy): {acc1:.4f} ({acc1*100:.2f}%)\n")
        f.write(f"ACC5 (Top-5 Accuracy): {acc5:.4f} ({acc5*100:.2f}%)\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(test_report)
    
    plot_training_history(history, run_id)
    torch.save(model.state_dict(), f'{run_id}.pth')
    
    total_time = time.time() - total_start_time
    print(f'\n✅ Model saved!')
    print(f'✅ {run_id} training completed!')
    print(f'Total time: {total_time/3600:.2f} hours')


if __name__ == '__main__':
    main()
