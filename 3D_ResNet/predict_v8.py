import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
import time
import collections
from typing import Optional, Tuple, Dict


# =========================================================
# SAFE MODEL LOADER (ENV-PROOF)
# =========================================================
def _get_r3d_transform_from_weights(weights=None):
    # R3D 模型的標準規範：Kinetics-400 資料集統計出來的影片平均值與標準差
    mean = [0.43216, 0.394666, 0.37645]
    std  = [0.22803, 0.22145, 0.216989]
    # 如果權重物件有 meta 資訊，且包含 mean/std，則使用它們
    if weights is not None:
        try:
            meta = weights.meta
            if "mean" in meta and "std" in meta:
                mean = meta["mean"]
                std = meta["std"]
        except Exception:
            pass

    transform = transforms.Compose([
        transforms.Resize((128, 171)), # R3D 的標準輸入尺寸是 128x171，這裡先調整到這個尺寸，然後再切出 112x112 的正方形；如果直接調整到 224x224，可能會改變影片的寬高比，導致預測不準確。
        # transforms.Resize((1280, 171)),
        transforms.CenterCrop((112, 112)), #切出 112X1122的 正方形
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std) # R3D 的標準化參數
    ])
    return transform


def _safe_load_state_dict(model: nn.Module, ckpt_path: str, num_classes: int):
    # 怕沒有 GPU 或 torchvision 版本問題，導致模型或權重無法載入，這裡做一層保護。
    state = torch.load(ckpt_path, map_location="cpu")

    # 嘗試從常見的 checkpoint 格式中提取 state_dict
    # 提取權重的邏輯：優先 "state_dict"，其次 "model_state_dict"，最後假設整個檔案就是 state_dict
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    #檢查 LAYER 4 的 FC 是否跟我們的CLASS數量匹配
    if "fc.weight" in state:
        w = state["fc.weight"]
        if hasattr(w, "shape") and w.shape[0] != num_classes:
            raise RuntimeError(
                f"Checkpoint fc.weight shape {tuple(w.shape)} != expected ({num_classes}, ...).\n"
                f"Your class_names order/count is NOT matching training.\n"
                f"Fix class_names to exactly match training."
            )

    missing, unexpected = model.load_state_dict(state, strict=False)

    #判斷如果 fc.weight 沒有被載入，這通常意味著 checkpoint 的 head 與我們的模型不匹配
    if "fc.weight" in missing: # 
        raise RuntimeError(
            "fc.weight was NOT loaded from checkpoint (missing). "
            "This means your checkpoint doesn't match this model head. "
            "Stop here, fix class_names / checkpoint / architecture."
        )

    return missing, unexpected


def load_model_env_proof(model_path: str, num_classes: int) -> Tuple[nn.Module, transforms.Compose]:
    """
    Robust loader:
    - Prefer r3d_18 with pretrained weights
    - If weights download/import fails -> r3d_18(weights=None)
    - If torchvision.video totally broken -> try ResNet18 fallback (ONLY if checkpoint fits)
    """
    # ---------- Try R3D ----------
    try:
        from torchvision.models.video import r3d_18, R3D_18_Weights
        weights = None
        try:
            #官方預訓練權重，會自動下載（需要網路），如果失敗就退回不使用權重的版本
            weights = R3D_18_Weights.DEFAULT
            model = r3d_18(weights=weights)
            transform = _get_r3d_transform_from_weights(weights)
        except Exception:
            # 空的權重物件，或下載失敗，都退回不使用權重的版本
            model = r3d_18(weights=None)
            transform = _get_r3d_transform_from_weights(None)

        #將最後一層全連接層改成我們的類別數量
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # 嘗試載入 checkpoint，這裡會檢查 fc 層是否匹配
        _safe_load_state_dict(model, model_path, num_classes)
        model.eval()
        return model, transform

    except Exception as e_r3d:
        print(f"[WARN] R3D load failed: {e_r3d}")

    # ---------- Fallback: ResNet18 adapted to 3D (VERY RISKY) ----------
    # 將 ResNet18 的 conv1 改成 3D 卷積，並調整參數以適應影片輸入
    try: 
        from torchvision.models import resnet18, ResNet18_Weights

        # Build model
        try:
            weights = ResNet18_Weights.DEFAULT # 官方預訓練權重 ResNet18-2D
            model = resnet18(weights=weights)
        except Exception:
            model = resnet18(weights=None)

        # 將 ResNet18 的 conv1 改成 3D 卷積，並調整參數以適應影片輸入
        model.conv1 = nn.Conv3d(
            3, 64, kernel_size=(3, 7, 7),
            stride=(1, 2, 2), padding=(1, 3, 3), bias=False
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes) 

        # ResNet18 的標準化參數（ImageNet 2D）
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # 嘗試載入 checkpoint，這裡會檢查 fc 層是否匹配；如果 checkpoint 是為 R3D 訓練的，這裡很可能會因為 conv1 的權重不匹配而失敗，這時候就會捕捉到錯誤並報告。
        _safe_load_state_dict(model, model_path, num_classes)
        model.eval()
        print("[WARN] Using ResNet18-3D fallback. Accuracy may be unreliable vs training.")
        return model, transform

    except Exception as e_res:
        raise RuntimeError(
            f"Failed to load any usable model. Last error: {e_res}"
        )


# =========================================================
# 已完成的動作
# =========================================================
def draw_right_text(img, text, y, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7,
                    color=(255, 255, 255), thickness=2, margin=20):
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = img.shape[1] - tw - margin
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

# =========================================================
# 尚未完成的動作
# =========================================================


def draw_left_text(img, text, y, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7,
                   color=(255, 255, 255), thickness=2, margin=20):
    x = margin
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# =========================================================
# 相機預測主程式
# =========================================================
def predict_live_camera(camera_index, model, transform, class_names, clip_len=16, device="cpu"):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Cannot open video/camera:", camera_index)
        
        return
    
    # 建立相機視窗
    win_name = "Action Validation"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    #讀取影片的幀率，如果讀取失敗則預設為30FPS
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps is None or input_fps <= 1:
        input_fps = 30.0  # safe fallback

    # 根據幀率和剪輯長度計算每個剪輯的持續時間
    clip_duration = clip_len / input_fps # 影片讀取16FPS 在我們實際影片是10FPS的情況下，這裡的持續時間會是1.6秒，代表每次預測都是基於過去1.6秒的影片內容。
    min_stay_time = 3.0 #動作持續 3 秒
    conf_threshold = 0.6 # 信心值超過 0.6

    # 建立類別狀態字典，追蹤每個類別的持續時間和完成狀態
    hidden_classes = {"Take", "Screwin"}  # 這裡是我們不顯示在左側待辦清單中的類別，因為它們不需要持續驗證
    display_classes = [c for c in class_names if c not in hidden_classes] # 顯示在左側待辦清單中的類別
    class_states = {name: {"active_time": 0.0, "is_ok": False} for name in display_classes} # 追蹤待辦類別的狀態（持續時間和是否完成）

    # 儲存影片
    save_video = True
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_path = "output_result_envproof3.avi"
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640 
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        writer = cv2.VideoWriter(out_path, fourcc, input_fps, (width, height)) # 使用輸入影片的幀率來儲存，確保同步

    frame_buffer = collections.deque(maxlen=clip_len) # 用來存放當前剪輯的幀，當達到 clip_len 時就進行預測

    model = model.to(device)
    model.eval()

    print("▶ START – Press Q to quit")

    last_pred_class = "Waiting" # 預設初始狀態為 "Waiting"，直到模型給出第一個預測結果
    last_conf = 0.0 
    last_infer_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 將當前幀加入剪輯緩衝區    
        frame_buffer.append(frame.copy())

        if len(frame_buffer) == clip_len: #當緩衝區達到16幀時，進行預測
            start_t = time.time()

            clip_tensor = []
            for f in frame_buffer:
                # 對這16幀進行與訓練相同的預處理，轉換成模型輸入的格式
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                clip_tensor.append(transform(pil))
            # 將16幀的張量堆疊成一個4D張量，形狀為 (1, 3, 16, H, W)，準備輸入模型進行預測
            input_tensor = torch.stack(clip_tensor, dim=1).unsqueeze(0).to(device)

            #開始預測，並計算推論時間和FPS
            with torch.no_grad():
                logits = model(input_tensor) # 模型輸出原始分數（logits），形狀為 (1, num_classes)
                prob = torch.softmax(logits, dim=1) # 對 logits 進行 softmax，得到每個類別的預測概率
                conf, pred = torch.max(prob, 1) # 找到概率最高的類別和對應的信心值（confidence）

            last_pred_class = class_names[pred.item()]
            last_conf = conf.item()

            # 根據預測結果更新類別狀態：如果當前預測的類別在待辦清單中，且信心值超過門檻，則開始累積該類別的持續時間；如果持續時間達到要求，則標記為完成；如果當前預測的類別不是該類別，則重置該類別的持續時間。
            if (last_conf >= conf_threshold) and (last_pred_class in class_states):
                for name in class_states:
                    if name == last_pred_class:
                        class_states[name]["active_time"] += clip_duration 
                        if class_states[name]["active_time"] >= min_stay_time: 
                            class_states[name]["is_ok"] = True
                    else:
                        class_states[name]["active_time"] = 0.0

            proc_time = time.time() - start_t
            last_infer_fps = (clip_len / proc_time) if proc_time > 0 else 0.0

        # 在畫面上顯示當前預測的類別、信心值和推論FPS，以及左側的待辦清單和右側的已完成清單
        cv2.putText(frame, f"Infer FPS: {last_infer_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Pred: {last_pred_class} ({last_conf:.2f})", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        # 左側面板顯示待辦活動，只有當前未完成的類別會顯示在這裡，並且以紅色標示
        y_left = 110
        draw_left_text(frame, "PENDING ACTIVITIES", y_left, scale=0.55, color=(0, 255, 255), thickness=1)
        y_left += 35
        for name in display_classes:
            if not class_states[name]["is_ok"]:
                draw_left_text(frame, f"{name}: n/a", y_left, scale=0.5, color=(0, 0, 255), thickness=1)
                y_left += 28

        # 右側面板顯示已完成的活動，只有當前已完成的類別會顯示在這裡，並且以綠色標示；如果沒有任何類別完成，則右側面板會顯示 "COMPLETED" 的標題，但不會列出任何類別。
        y_right = 110
        draw_right_text(frame, "COMPLETED", y_right, scale=0.55, color=(0, 255, 0), thickness=1)
        y_right += 35
        for name in display_classes:
            if class_states[name]["is_ok"]:
                draw_right_text(frame, f"{name}: done", y_right, scale=0.5, color=(0, 255, 0), thickness=1)
                y_right += 28

        # 將處理後的畫面顯示在視窗中，並且如果需要儲存影片，則將當前幀寫入影片檔案；同時監聽鍵盤事件，如果按下 "Q" 鍵則退出循環。
        display = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(win_name,display)

        if save_video and writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if save_video and writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("■ STOP")


# =========================================================
# MAIN
# =========================================================
def main():
    # model_path = "run2.pth"
    model_path ="run0115_finetune_EDA.pth"

    # MUST match training class order exactly
    class_names = [
        "Assemble1",
        "Assemble2",
        "Assemble3",
        "Brush",
        "Screwin",
        "Take",
        "Waiting"
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model, transform = load_model_env_proof(model_path, len(class_names))

    predict_live_camera(
        # camera_index="dataset/2026/0114/a19.mp4",  # or 0 webcam
        camera_index=0,  # or 0 webcam
        model=model,
        transform=transform,
        class_names=class_names,
        clip_len=16, ## taking 16 frame to predict (can change to 8 for smooth)
        device=device,
    )


if __name__ == "__main__":
    main()
