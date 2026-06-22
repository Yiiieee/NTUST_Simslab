import os
import time
import collections
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
import torchvision.transforms as transforms
from torch.cuda.amp import autocast
from tqdm import tqdm  


# UI 繪圖輔助函數

def draw_right_text(img, text, y, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7,
                    color=(255, 255, 255), thickness=2, margin=20):
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = img.shape[1] - tw - margin
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_left_text(img, text, y, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7,
                   color=(255, 255, 255), thickness=2, margin=20):
    x = margin
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# 1. 模型載入與前處理

def _safe_load_state_dict(model: nn.Module, ckpt_path: str, num_classes: int):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)

def load_r3d_model(model_path: str, num_classes: int, device: str) -> nn.Module:
    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    _safe_load_state_dict(model, model_path, num_classes)
    model = model.to(device)
    model.eval()
    if int(torch.__version__.split('.')[0]) >= 2:
        model = torch.compile(model)
    return model

def get_gpu_transform(device):
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1).to(device)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1).to(device)
    return nn.Sequential(
        transforms.Resize((128, 171), antialias=True),
        transforms.CenterCrop((112, 112)),
        transforms.Normalize(mean=mean, std=std)
    ).to(device)


# 第一階段：批次推論 (產出時間軸預測字典)

def analyze_video_pass1(video_path, model, class_names, device, batch_size=32, clip_len=16, infer_stride=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_buffer = collections.deque(maxlen=clip_len)
    batch_clips = []
    batch_timestamps = []
    gpu_transform = get_gpu_transform(device)
    
    predictions_dict = {} 
    current_frame = 0

    print(f"\n[階段 1/2] GPU (Batch={batch_size})...")
    pbar = tqdm(total=total_frames, desc="分析進度", unit="幀")

    while True:
        ret, frame = cap.read()
        
        # 如果有成功讀取到畫面，就塞入 Buffer
        if ret:
            current_frame += 1
            pbar.update(1)
            frame_buffer.append(frame)

            if len(frame_buffer) == clip_len and current_frame % infer_stride == 0:
                batch_clips.append(list(frame_buffer))
                batch_timestamps.append(current_frame)

        #  修復尾刀Bug：只要滿一個 Batch，或者影片結束但還有剩餘資料，就 GPU 處理
        if len(batch_clips) == batch_size or (not ret and len(batch_clips) > 0):
            np_batch = np.array(batch_clips)[..., ::-1].copy() # BGR to RGB
            tensor_batch = torch.from_numpy(np_batch).float() / 255.0
            
            B = tensor_batch.shape[0]
            tensor_batch = tensor_batch.permute(0, 1, 4, 2, 3).contiguous()
            tensor_batch = tensor_batch.view(B * clip_len, 3, tensor_batch.shape[3], tensor_batch.shape[4])
            tensor_batch = tensor_batch.to(device, non_blocking=True)
            
            processed_batch = gpu_transform(tensor_batch) 
            input_tensor = processed_batch.view(B, clip_len, 3, 112, 112).permute(0, 2, 1, 3, 4)

            with torch.no_grad():
                with autocast():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1)
                    confs, preds = torch.max(probs, 1)

            confs = confs.cpu().numpy()
            preds = preds.cpu().numpy()
            
            for i in range(B):
                frame_idx = batch_timestamps[i]
                predictions_dict[frame_idx] = {
                    "action": class_names[preds[i]],
                    "confidence": confs[i]
                }
            
            # 處理完清空，準備裝下一批
            batch_clips = []
            batch_timestamps = []

        
        if not ret:
            break

    pbar.close()
    cap.release()
    
    # 強制清空 GPU 暫存，避免 Ubuntu OOM Killed 直接跳掉程式
    if device == "cuda":
        torch.cuda.empty_cache()
        
    return predictions_dict


# 第二階段：渲染影片與狀態判定 

def render_video_pass2(video_path, out_path, predictions_dict, class_names):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
   
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print(f"\n 錯誤：無法建立影片檔案！請確認路徑或權限: {out_path}")
        return
    
    #  任務狀態設定
    min_stay_time = 1.0   # 動作真實累積滿 1.0 秒即判定完成
    conf_threshold = 0.5  # 信心門檻
    max_patience = int(fps * 0.5) # 容忍度：允許大約 0.5 秒的預測失誤不中斷
    
    hidden_classes = {"Take"}
    display_classes = [c for c in class_names if c not in hidden_classes]
    
    class_states = {
        name: {
            "active_time": 0.0, 
            "is_ok": False,
            "patience": 0 
        } for name in display_classes
    }
    
    current_frame = 0
    current_pred = "Waiting"
    current_conf = 0.0
    frame_duration = 1.0 / fps

    print(f"\n[階段 2/2] 繪製 UI 並儲存影片 ( {fps:.2f} FPS)...")
    pbar = tqdm(total=total_frames, desc="輸出進度", unit="幀")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame += 1
        pbar.update(1)
        
        if current_frame in predictions_dict:
            current_pred = predictions_dict[current_frame]["action"]
            current_conf = predictions_dict[current_frame]["confidence"]

        # 每個動作獨立判斷
        for name in class_states:
            # 如果目前這個畫面判斷是這個動作，且信心達標
            if name == current_pred and current_conf >= conf_threshold:
                class_states[name]["active_time"] += frame_duration
                class_states[name]["patience"] = max_patience # 補滿容忍度
                
                if class_states[name]["active_time"] >= min_stay_time:
                    class_states[name]["is_ok"] = True
            else:
                # 如果判斷成別的動作，或是信心太低 -> 開始扣容忍度
                if class_states[name]["patience"] > 0:
                    class_states[name]["patience"] -= 1
                else:
                    # 容忍度扣光了，累積時間歸零
                    class_states[name]["active_time"] = 0.0

        # --- 畫畫面 UI ---
        cv2.putText(frame, f"Pred: {current_pred} ({current_conf:.2f})", (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        y_left = 80
        draw_left_text(frame, "PENDING ACTIVITIES", y_left, scale=0.75, color=(0, 255, 255), thickness=2)
        y_left += 35
        for name in display_classes:
            if not class_states[name]["is_ok"]:
                draw_left_text(frame, f"{name}: NOT OK", y_left, scale=0.7, color=(0, 0, 255), thickness=2)
                y_left += 28

        y_right = 80
        draw_right_text(frame, "COMPLETED", y_right, scale=0.75, color=(0, 255, 0), thickness=2)
        y_right += 35
        for name in display_classes:
            if class_states[name]["is_ok"]:
                draw_right_text(frame, f"{name}: DONE", y_right, scale=0.7, color=(0, 255, 0), thickness=2)
                y_right += 28

        writer.write(frame)

    pbar.close()
    cap.release()
    writer.release()
    print(f"\n✓ 成功儲存影片至: {out_path}")


# 主程式

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # =========================================================
    # 路徑設定
    model_path = os.path.join(script_dir, "run0115_finetune_EDA_byZiv.pth")   
    video_path = os.path.join(script_dir, "a5.mp4")
    # =========================================================
    
    out_path = os.path.join(script_dir, "output_result_1x_speed_byZiv.mp4")

    class_names = ["Assemble1", "Assemble2", "Assemble3", "Brush", "Screwin", "Take", "Waiting"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用硬體: {device}")

    try:
        model = load_r3d_model(model_path, len(class_names), device)

        predictions_dict = analyze_video_pass1(
            video_path=video_path,
            model=model,
            class_names=class_names,
            device=device,
            batch_size=16,      
            clip_len=16,       
            infer_stride=8     
        )

        render_video_pass2(
            video_path=video_path,
            out_path=out_path,
            predictions_dict=predictions_dict,
            class_names=class_names
        )
    except Exception as e:
        print(f"\n發生錯誤導致中止: {e}")

if __name__ == "__main__":
    main()