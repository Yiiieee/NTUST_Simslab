import torch
from torch import optim


def build_optimizer(cfg, model, base_lr=0.0, resume=None):
    print('==============================')
    print('Optimizer: {}'.format(cfg['optimizer']))
    print('--momentum: {}'.format(cfg['momentum']))
    print('--weight_decay: {}'.format(cfg['weight_decay']))

    if cfg['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=base_lr,
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'])

    elif cfg['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=base_lr,
            eight_decay=cfg['weight_decay'])
                                
    elif cfg['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=base_lr,
            weight_decay=cfg['weight_decay'])
          
    start_epoch = 0
    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume, map_location='cpu')
        
        try:
            # 嘗試讀取 optimizer
            checkpoint_state_dict = checkpoint.pop("optimizer")
            optimizer.load_state_dict(checkpoint_state_dict)
            start_epoch = checkpoint.pop("epoch")
            print(f"成功恢復 Optimizer 狀態，將從 Epoch {start_epoch} 繼續訓練。")
        except KeyError:
            # 如果預訓練權重裡沒有 optimizer 狀態（遷移學習常見情況）
            print("警告: 預訓練權重中未找到 'optimizer' 狀態，這在遷移學習中是正常的。")
            print("將使用全新的 Optimizer，並從 Epoch 0 開始訓練。")
            start_epoch = 0
                        
                                
    return optimizer, start_epoch
