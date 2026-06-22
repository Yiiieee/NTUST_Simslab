import os
import threading
import queue
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
import time
import collections
from typing import Tuple
from tqdm import tqdm

# =========================================================
# SAFE MODEL LOADER
# =========================================================
def _get_r3d_transform_from_weights(weights=None):
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]
    if weights is not None:
        try:
            meta = weights.meta
            if "mean" in meta and "std" in meta:
                mean = meta["mean"]
                std = meta["std"]
        except Exception: pass
    return transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def _safe_load_state_dict(model: nn.Module, ckpt_path: str, num_classes: int):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if "fc.weight" in state:
        w = state["fc.weight"]
        if hasattr(w, "shape") and w.shape[0] != num_classes:
            raise RuntimeError(f"Checkpoint shape {tuple(w.shape)} mismatch.")
    model.load_state_dict(state, strict=False)
    return

def load_model_env_proof(model_path: str, num_classes: int) -> Tuple[nn.Module, transforms.Compose]:
    try:
        from torchvision.models.video import r3d_18, R3D_18_Weights
        try:
            weights = R3D_18_Weights.DEFAULT
            model = r3d_18(weights=weights)
            transform = _get_r3d_transform_from_weights(weights)
        except:
            model = r3d_18(weights=None)
            transform = _get_r3d_transform_from_weights(None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        _safe_load_state_dict(model, model_path, num_classes)
        model.eval()
        return model, transform
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")

# =========================================================
# UI HELPERS
# =========================================================
def draw_right_text(img, text, y, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7, color=(255, 255, 255), thickness=2):
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (img.shape[1] - tw - 20, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_left_text(img, text, y, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, (20, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_center_alert_transparent(frame, text, bg_color=(0, 0, 255), alpha=0.78):
    h, w, _ = frame.shape
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    x, y = (w - tw) // 2, (h + th) // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x-40, y-th-24), (x+tw+40, y+baseline+24), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    cv2.putText(frame, text, (x, y), font, scale, (255,255,255), thick, cv2.LINE_AA)

# =========================================================
# PREDICTION LOGIC 
# =========================================================
def predict_video_file(video_path, out_path, model, transform, class_names, clip_len=16, device="cpu", pbar_desc=""):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(out_path, fourcc, input_fps, (width, height))

    frame_delay = 1.0 / input_fps 
    min_stay_time = 3.0
    waiting_stay_time = 2.0  # wating時間
    conf_threshold = 0.6
    
    hidden_classes = {"Take"}
    display_classes = [c for c in class_names if c not in hidden_classes]
    class_states = {name: {"active_time": 0.0, "is_ok": False} for name in display_classes}

    infer_result = {"pred_class": "Waiting", "conf": 0.0, "infer_fps": 0.0}
    result_lock = threading.Lock()
    infer_queue = queue.Queue(maxsize=2)
    alert_state = {"current_alert_text": None, "step_started": {"Assemble2": False, "Assemble3": False}, "detect_count": {"Assemble2": 0, "Assemble3": 0}}

    completion_times = {}
    current_frame_count = 0

    def infer_worker():
        while True:
            item = infer_queue.get()
            if item is None: break
            clip_tensor = [transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in item]
            input_tensor = torch.stack(clip_tensor, dim=1).unsqueeze(0).to(device)
            start_t = time.time()
            with torch.no_grad():
                prob = torch.softmax(model(input_tensor), dim=1)
                conf, pred = torch.max(prob, 1)
            elapsed = time.time() - start_t
            with result_lock:
                infer_result.update({"pred_class": class_names[pred.item()], "conf": conf.item(), "infer_fps": (clip_len/elapsed if elapsed>0 else 0)})
            infer_queue.task_done()

    worker_thread = threading.Thread(target=infer_worker, daemon=True)
    worker_thread.start()

    frame_buffer = collections.deque(maxlen=clip_len)
    frames_since_last_infer = 0
    
    pbar = tqdm(total=total_frames, desc=pbar_desc, unit="幀", leave=False, dynamic_ncols=True)

    while True:
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret: break

        current_frame_count += 1
        current_video_time = current_frame_count / input_fps

        frame_buffer.append(frame.copy())
        frames_since_last_infer += 1
        pbar.update(1)

        if len(frame_buffer) == clip_len and frames_since_last_infer >= clip_len:
            try:
                infer_queue.put_nowait(list(frame_buffer))
                frames_since_last_infer = 0
            except queue.Full: pass

        with result_lock:
            raw_pred, last_conf, last_fps = infer_result["pred_class"], infer_result["conf"], infer_result["infer_fps"]

        # =========================================================
        # 🛡️ 狀態機
        # =========================================================
        final_pred = raw_pred
        is_a2_ok = class_states.get("Assemble2", {}).get("is_ok", False)
        a2_done_time = completion_times.get("Assemble2", 0.0)
        
        if raw_pred == "Assemble3":
            if not is_a2_ok:
                # 規則 1：偵測到 2 之前，絕對不允許 3 出現
                final_pred = "Assemble2"
            else:
                # 規則 2：2 跟 3 之間必須間隔超過 ? 秒
                if (current_video_time - a2_done_time) <= 5.0:
                    final_pred = "Waiting" # 強制進入冷卻期
                    
        elif raw_pred == "Assemble2":
            # 規則 3：若目前理應是組裝 3 的階段 (2已完成且過了 ? 秒冷卻)，但模型猜成 2，直接把它視為 3
            if is_a2_ok and (current_video_time - a2_done_time) > 5.0: 
                final_pred = "Assemble3"
        # =========================================================

        # Temporal Validation (時間累加邏輯)
        if last_conf >= conf_threshold and final_pred in class_states:
            for name in class_states:
                if name == final_pred:
                    class_states[name]["active_time"] += frame_delay 
                    
                    # 🌟 判斷需要的達標時間：Waiting 其他維持 (3秒)
                    req_time = waiting_stay_time if name == "Waiting" else min_stay_time
                    
                    if class_states[name]["active_time"] >= req_time: 
                        if not class_states[name]["is_ok"]:
                            class_states[name]["is_ok"] = True
                            completion_times[name] = current_video_time 
                else: 
                    class_states[name]["active_time"] = 0.0

        a = alert_state
        if class_states.get("Assemble1",{}).get("is_ok") and not a["step_started"]["Assemble2"]: a["current_alert_text"] = "ROBOT PREPARING PRODUCT (ASSEMBLE 2)"
        if final_pred == "Assemble2" and last_conf >= conf_threshold: a["detect_count"]["Assemble2"] += 1
        else: a["detect_count"]["Assemble2"] = 0
        if a["detect_count"]["Assemble2"] >= 2:
            a["step_started"]["Assemble2"] = True
            if a["current_alert_text"] == "ROBOT PREPARING PRODUCT (ASSEMBLE 2)": a["current_alert_text"] = None

        # =========================================================
        # UI Drawing 
        # =========================================================
        cv2.putText(frame, f"Infer FPS: {last_fps:.2f}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 顯示狀態機的校正結果
        if raw_pred != final_pred:
            if final_pred == "Waiting" and raw_pred == "Assemble3":
                cv2.putText(frame, f"Pred: {final_pred} (Cooldown 3s)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            else:
                cv2.putText(frame, f"Pred: {final_pred} (Fixed from {raw_pred})", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        else:
            cv2.putText(frame, f"Pred: {final_pred} ({last_conf:.2f})", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        if a["current_alert_text"] and int(time.time() * 2) % 2 == 0:
            draw_center_alert_transparent(frame, a["current_alert_text"])

        # Panels (SOP 任務面板)
        y_l, y_r = 110, 110 
        draw_left_text(frame, "PENDING", y_l, color=(0, 255, 255)); y_l+=35
        draw_right_text(frame, "COMPLETED", y_r, color=(0, 255, 0)); y_r+=35
        for name in display_classes:
            if not class_states[name]["is_ok"]:
                draw_left_text(frame, f"{name}: NO", y_l, color=(0, 0, 255)); y_l+=28
            else:
                draw_right_text(frame, f"{name}: DONE", y_r, color=(0, 255, 0)); y_r+=28

        # Display & Save
        writer.write(frame)
        
        elapsed = time.time() - loop_start
        if cv2.waitKey(max(1, int((frame_delay-elapsed)*1000))) & 0xFF == ord("q"): break

    pbar.close()
    infer_queue.put(None)
    worker_thread.join()
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

# =========================================================
# MAIN 
# =========================================================
def main():
    model_path = "MoreAngle_EDA.pth"
    base_dir = "/home/simslab/Desktop/api/predict/test_0430/data"
    result_dir = "/home/simslab/Desktop/api/predict/test_0430/result_MoreAngle"
    
    if not os.path.exists(base_dir):
        print(f"❌ 找不到輸入目錄: {base_dir}")
        return
    os.makedirs(result_dir, exist_ok=True)

    class_names = ["Assemble1", "Assemble2", "Assemble3", "Brush", "Screwin", "Take", "Waiting"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    model, transform = load_model_env_proof(model_path, len(class_names))
    model.to(device)

    # 1. 取得 test 資料夾下所有影片檔案
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
    video_files = [f for f in os.listdir(base_dir) if f.lower().endswith(video_extensions)]
    video_files.sort() # 排序確保順序一致

    if not video_files:
        print("⚠️ test 資料夾內沒有發現影片檔案。")
        return

    print(f"📂 找到 {len(video_files)} 個影片，準備開始處理...")

    total_pbar = tqdm(video_files, desc="[總進度]", unit="部", dynamic_ncols=True)

    for video_filename in total_pbar:
        video_path = os.path.join(base_dir, video_filename)
        file_base = os.path.splitext(video_filename)[0]
        out_path = os.path.join(result_dir, f"{file_base}_moreAngle.avi")

        total_pbar.set_description(f"正在處理: {video_filename}")

        predict_video_file(
            video_path=video_path,
            out_path=out_path,
            model=model,
            transform=transform,
            class_names=class_names,
            clip_len=16,
            device=device,
            pbar_desc=f"影片進度"
        )

    print("\n✅ 所有影片處理完成。結果儲存於:", result_dir)

if __name__ == "__main__":
    main()