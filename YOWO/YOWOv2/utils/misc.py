import os
import torch
import torch.nn as nn

from dataset.ucf_jhmdb import UCF_JHMDB_Dataset
from dataset.ava import AVA_Dataset
from dataset.transforms import Augmentation, BaseTransform

from evaluator.ucf_jhmdb_evaluator import UCF_JHMDB_Evaluator
from evaluator.ava_evaluator import AVA_Evaluator




def build_dataset(d_cfg, args, is_train=False): #哪一組資料來訓練
    """
        d_cfg: dataset config
    """
    # 影像增強與預處理設定
    augmentation = Augmentation(
        img_size=d_cfg['train_size'],
        jitter=d_cfg['jitter'],
        hue=d_cfg['hue'],
        saturation=d_cfg['saturation'],
        exposure=d_cfg['exposure']
        )
    basetransform = BaseTransform(
        img_size=d_cfg['test_size'],
        )

    # --- 資料集判斷邏輯 ---
    
    # 1. 原有的 UCF24 與 JHMDB21
    if args.dataset in ['ucf24', 'jhmdb21']:
        data_dir = os.path.join(args.root, 'ucf24')

        dataset = UCF_JHMDB_Dataset(
            data_root=data_dir,
            dataset=args.dataset,
            img_size=d_cfg['train_size'],
            transform=augmentation,
            is_train=is_train,
            len_clip=args.len_clip,
            sampling_rate=d_cfg['sampling_rate']
            )
        num_classes = dataset.num_classes

        evaluator = UCF_JHMDB_Evaluator(
            data_root=data_dir,
            dataset=args.dataset,
            model_name=args.version,
            metric='fmap',
            img_size=d_cfg['test_size'],
            len_clip=args.len_clip,
            batch_size=args.test_batch_size,
            conf_thresh=0.01,
            iou_thresh=0.5,
            gt_folder=d_cfg['gt_folder'],
            save_path='./evaluator/eval_results/',
            transform=basetransform,
            collate_fn=CollateFunc()            
        )

    # 2. 你的自定義流水線數據集 (my_pipeline)
    elif args.dataset == 'my_pipeline':
        data_dir = args.root 

        # --- 【關鍵修復】強制讓這個類別的所有實例都認得類別數 ---
        # 這樣評估器內部建立的 testset 也會繼承到這個 7 (或你 config 設的數量)
        UCF_JHMDB_Dataset.num_classes = d_cfg['valid_num_classes'] 

        dataset = UCF_JHMDB_Dataset(
            data_root=data_dir,
            dataset='ucf24', # 借用 ucf24 的結構讀取 rgb/ 與 annotations/
            img_size=d_cfg['train_size'],
            transform=augmentation,
            is_train=is_train,
            len_clip=args.len_clip,
            sampling_rate=d_cfg['sampling_rate']
            )
        
        # 這裡也要確保變數有拿到值
        num_classes = dataset.num_classes

        # 初始化評估器
        evaluator = UCF_JHMDB_Evaluator(
            data_root=data_dir,
            dataset='ucf24', # 這裡同樣維持 ucf24，讓它走標準評估流程
            model_name=args.version,
            metric='fmap',
            img_size=d_cfg['test_size'],
            len_clip=args.len_clip,
            batch_size=args.test_batch_size,
            conf_thresh=0.01,
            iou_thresh=0.5,
            gt_folder=d_cfg['gt_folder'],
            save_path='./evaluator/eval_results/',
            transform=basetransform,
            collate_fn=CollateFunc()            
        )

    # 3. 原有的 AVA 資料集
    elif args.dataset == 'ava_v2.2':
        data_dir = os.path.join(args.root, 'AVA_Dataset')
        
        dataset = AVA_Dataset(
            cfg=d_cfg,
            data_root=data_dir,
            is_train=True,
            img_size=d_cfg['train_size'],
            transform=augmentation,
            len_clip=args.len_clip,
            sampling_rate=d_cfg['sampling_rate']
        )
        num_classes = 80

        evaluator = AVA_Evaluator(
            d_cfg=d_cfg,
            data_root=data_dir,
            img_size=d_cfg['test_size'],
            len_clip=args.len_clip,
            sampling_rate=d_cfg['sampling_rate'],
            batch_size=args.test_batch_size,
            transform=basetransform,
            collate_fn=CollateFunc(),
            full_test_on_val=False,
            version='v2.2'
            )

    else:
        print(f'unknow dataset !! "{args.dataset}" is not supported.')
        exit(0)

    print('==============================')
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))
    print('Number of classes:', num_classes)

    if not args.eval:
        evaluator = None

    return dataset, evaluator, num_classes


def build_dataloader(args, dataset, batch_size, collate_fn=None, is_train=False): #負責批次載入資料
    if is_train:
        if args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.RandomSampler(dataset)

        batch_sampler_train = torch.utils.data.BatchSampler(sampler, 
                                                            batch_size, 
                                                            drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, 
            batch_sampler=batch_sampler_train,
            collate_fn=collate_fn, 
            num_workers=args.num_workers,
            pin_memory=True
            )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, 
            shuffle=False,
            collate_fn=collate_fn, 
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=True
            )
    
    return dataloader
    

def load_weight(model, path_to_ckpt=None): # 載入權重
    if path_to_ckpt is None:
        print('No trained weight ..')
        return model
        
    checkpoint = torch.load(path_to_ckpt, map_location='cpu')
    checkpoint_state_dict = checkpoint.pop("model")
    model_state_dict = model.state_dict()
    
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                checkpoint_state_dict.pop(k)
        else:
            checkpoint_state_dict.pop(k)
            print(f"Skipping weight: {k}")

    model.load_state_dict(checkpoint_state_dict)
    print('Finished loading model!')

    return model


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class CollateFunc(object): #打包批次資料的函數，將原本的 Tensor 轉為 Loss 函數預期的字典格式
    def __call__(self, batch):
        batch_frame_id = []
        batch_video_clips = []
        batch_targets = []

        for sample in batch:
            batch_frame_id.append(sample[0])
            batch_video_clips.append(sample[1])
            
            # --- 關鍵修正：將 Tensor 轉為字典格式 ---
            target = sample[2] # 這是 [N, 5] 的 Tensor: [x1, y1, x2, y2, cls_id]
            
            # 建立 Loss 函數預期的字典結構
            target_dict = {
                'boxes': target[:, :4],         # 取前四列作為座標
                'labels': target[:, 4].long()   # 取最後一列並轉為長整數作為類別
            }
            batch_targets.append(target_dict)
            # ------------------------------------

        # 將影片序列堆疊成 [B, 3, T, H, W]
        batch_video_clips = torch.stack(batch_video_clips)
        
        return batch_frame_id, batch_video_clips, batch_targets