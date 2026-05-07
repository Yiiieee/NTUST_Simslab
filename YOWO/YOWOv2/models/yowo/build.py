import torch
from .yowo import YOWO
from .loss import build_criterion


# build YOWO detector
def build_yowo(args,
                d_cfg,
                m_cfg, 
                device, 
                num_classes=80, 
                trainable=False,
                resume=None):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))

    # build YOWO
    model = YOWO(
        cfg = m_cfg,
        device = device,
        num_classes = num_classes,
        conf_thresh = args.conf_thresh,
        nms_thresh = args.nms_thresh,
        topk = args.topk,
        trainable = trainable,
        multi_hot = d_cfg['multi_hot'],
        )

    if trainable:
        # Freeze backbone
        if args.freeze_backbone_2d:
            print('Freeze 2D Backbone ...')
            for m in model.backbone_2d.parameters():
                m.requires_grad = False
        if args.freeze_backbone_3d:
            print('Freeze 3D Backbone ...')
            for m in model.backbone_3d.parameters():
                m.requires_grad = False
            
        # keep training       
        if resume is not None:
            print('keep training: ', resume)
            checkpoint = torch.load(resume, map_location='cpu')
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            #model.load_state_dict(checkpoint_state_dict)
            print("正在進行遷移學習權重過濾...")
            model_state_dict = model.state_dict()
            # 建立一個新的字典，只存放形狀匹配的權重
            filtered_state_dict = {}
            for k, v in checkpoint_state_dict.items():
                if k in model_state_dict:
                    # 檢查形狀是否一致
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        print(f"跳過層: {k} (原因: 形狀不符, 權重:{v.shape} vs 模型:{model_state_dict[k].shape})")
                else:
                    print(f"跳過層: {k} (原因: 模型中不存在此層)")

            # 載入過濾後的權重，strict 設為 False 允許部分載入
            model.load_state_dict(filtered_state_dict, strict=False)
            print("✅ 權重過濾載入完成！已保留 Backbone 權重，分類頭將重新初始化。")
            

        # build criterion
        criterion = build_criterion(
            args, d_cfg['train_size'], num_classes, d_cfg['multi_hot'])
    
    else:
        criterion = None
                        
    return model, criterion
