import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
import time
import collections
from typing import Tuple


# =========================================================
# SAFE MODEL LOADER (ENV-PROOF)
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
        except Exception:
            pass

    transform = transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform


def _safe_load_state_dict(model: nn.Module, ckpt_path: str, num_classes: int):
    state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    if "fc.weight" in state:
        w = state["fc.weight"]
        if hasattr(w, "shape") and w.shape[0] != num_classes:
            raise RuntimeError(
                f"Checkpoint fc.weight shape {tuple(w.shape)} != expected ({num_classes}, ...).\n"
                f"Your class_names order/count is NOT matching training."
            )

    missing, unexpected = model.load_state_dict(state, strict=False)

    if "fc.weight" in missing:
        raise RuntimeError(
            "fc.weight was NOT loaded from checkpoint. "
            "Fix class_names / checkpoint / architecture."
        )

    return missing, unexpected


def load_model_env_proof(model_path: str, num_classes: int) -> Tuple[nn.Module, transforms.Compose]:
    try:
        from torchvision.models.video import r3d_18, R3D_18_Weights

        try:
            weights = R3D_18_Weights.DEFAULT
            model = r3d_18(weights=weights)
            transform = _get_r3d_transform_from_weights(weights)
        except Exception:
            model = r3d_18(weights=None)
            transform = _get_r3d_transform_from_weights(None)

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        _safe_load_state_dict(model, model_path, num_classes)
        model.eval()
        return model, transform

    except Exception as e_r3d:
        print(f"[WARN] R3D load failed: {e_r3d}")

    try:
        from torchvision.models import resnet18, ResNet18_Weights

        try:
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
        except Exception:
            model = resnet18(weights=None)

        model.conv1 = nn.Conv3d(
            3, 64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        _safe_load_state_dict(model, model_path, num_classes)
        model.eval()
        print("[WARN] Using ResNet18-3D fallback. Accuracy may be unreliable.")
        return model, transform

    except Exception as e_res:
        raise RuntimeError(
            f"Failed to load any usable model. Last error: {e_res}"
        )


# =========================================================
# UI HELPERS
# =========================================================
def draw_right_text(img, text, y, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7,
                    color=(255, 255, 255), thickness=2, margin=20):
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = img.shape[1] - tw - margin
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_left_text(img, text, y, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7,
                   color=(255, 255, 255), thickness=2, margin=20):
    cv2.putText(img, text, (margin, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_center_alert_transparent(frame, text, bg_color=(0, 0, 255), text_color=(255, 255, 255),
                                  font_scale=1.4, thickness=3, pad_x=40, pad_y=24, alpha=0.78):
    h, w, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x = (w - tw) // 2
    y = (h + th) // 2

    top_left = (x - pad_x, y - th - pad_y)
    bottom_right = (x + tw + pad_x, y + baseline + pad_y)

    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


# =========================================================
# LIVE PREDICTION + TEMPORAL VALIDATION + ALERT LOGIC
# =========================================================
def predict_live_camera(camera_index, model, transform, class_names, clip_len=16, device="cpu"):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("❌ Cannot open video/camera:", camera_index)
        return

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps is None or input_fps <= 1:
        input_fps = 30.0

    # ===  系統與狀態機參數設定區 ===
    CONF_THRESHOLD = 0.6          # 模型預測的最低信心值門檻
    MIN_STAY_TIME_DEFAULT = 3.0   # 一般動作(如 Assemble1, 2, 3) 需要維持的秒數
    MIN_STAY_TIME_WAITING = 1.0   # Waiting 只需要維持的秒數 (較短)
    COOLDOWN_A2_TO_A3 = 5.0       # Assemble2 完成後，間隔幾秒才能被判定為 Assemble3
    DETECT_NEEDED = 2             # Alert 觸發需要的連續偵測次數
    # =================================================

    # 計算每一幀經歷的真實時間秒數
    frame_delay = 1.0 / input_fps 

    hidden_classes = {"Take"}
    display_classes = [c for c in class_names if c not in hidden_classes]

    class_states = {
        name: {"active_time": 0.0, "is_ok": False}
        for name in display_classes
    }

    save_video = True
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_path = "output_result_alert_assemble2_3.avi"
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        writer = cv2.VideoWriter(out_path, fourcc, input_fps, (width, height))

    frame_buffer = collections.deque(maxlen=clip_len)

    model = model.to(device)
    model.eval()

    print("▶ START – Press Q to quit")

    last_pred_class = "Waiting"
    last_conf = 0.0
    last_infer_fps = 0.0

    # 狀態機使用的計時與完成紀錄
    completion_times = {}
    current_frame_count = 0

    # =====================================================
    # ALERT STATE FOR ASSEMBLE 2 AND ASSEMBLE 3
    # =====================================================
    current_alert_text = None

    step_started = {
        "Assemble2": False,
        "Assemble3": False
    }

    detect_count = {
        "Assemble2": 0,
        "Assemble3": 0
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_count += 1
        current_video_time = current_frame_count * frame_delay

        frame_buffer.append(frame.copy())

        if len(frame_buffer) == clip_len:
            start_t = time.time()

            clip_tensor = []
            for f in frame_buffer:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                clip_tensor.append(transform(pil))

            input_tensor = torch.stack(clip_tensor, dim=1).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                prob = torch.softmax(logits, dim=1)
                conf, pred = torch.max(prob, 1)

            last_pred_class = class_names[pred.item()]
            last_conf = conf.item()

            proc_time = time.time() - start_t
            last_infer_fps = (clip_len / proc_time) if proc_time > 0 else 0.0

        # ==============================================
        # 狀態機邏輯 
        # ==============================================
        raw_pred = last_pred_class
        final_pred = raw_pred
        
        is_a2_ok = class_states.get("Assemble2", {}).get("is_ok", False)
        a2_done_time = completion_times.get("Assemble2", 0.0)

        if raw_pred == "Assemble3":
            if not is_a2_ok:
                # 規則 1：偵測到 2 之前 不會偵測到 3
                final_pred = "Assemble2"
            else:
                # 規則 2：2 跟 3 之間要間隔超過真實 COOLDOWN 秒數才會被判讀
                if (current_video_time - a2_done_time) <= COOLDOWN_A2_TO_A3:
                    final_pred = "Waiting" # 強制冷卻中
                    
        elif raw_pred == "Assemble2":
            # 規則 3：若再組裝 3 的時候(2已完成且過冷卻)偵測到 2，視為 3
            if is_a2_ok and (current_video_time - a2_done_time) > COOLDOWN_A2_TO_A3:
                final_pred = "Assemble3"

        # ==============================================
        # TEMPORAL VALIDATION (時間累加檢核)
        # ==============================================
        if (last_conf >= CONF_THRESHOLD) and (final_pred in class_states):
            for name in class_states:
                if name == final_pred:
                    # 使用 frame_delay 精準計算真實世界的秒數
                    class_states[name]["active_time"] += frame_delay
                    
                    # 決定需要的秒數 (Waiting 較短，其餘套用預設)
                    req_time = MIN_STAY_TIME_WAITING if name == "Waiting" else MIN_STAY_TIME_DEFAULT
                    
                    if class_states[name]["active_time"] >= req_time:
                        if not class_states[name]["is_ok"]:
                            class_states[name]["is_ok"] = True
                            completion_times[name] = current_video_time
                else:
                    class_states[name]["active_time"] = 0.0

        # ==============================================
        # ALERT LOGIC (基於 final_pred 觸發)
        # ==============================================
        if class_states.get("Assemble1", {}).get("is_ok", False) and not step_started["Assemble2"]:
            current_alert_text = "ROBOT PREPARING PRODUCT (ASSEMBLE 2)"

        if final_pred == "Assemble2" and last_conf >= CONF_THRESHOLD:
            detect_count["Assemble2"] += 1
        else:
            detect_count["Assemble2"] = 0

        if detect_count["Assemble2"] >= DETECT_NEEDED:
            step_started["Assemble2"] = True
            if current_alert_text == "ROBOT PREPARING PRODUCT (ASSEMBLE 2)":
                current_alert_text = None

        if class_states.get("Assemble2", {}).get("is_ok", False) and not step_started["Assemble3"]:
            current_alert_text = "ROBOT PREPARING PRODUCT (ASSEMBLE 3)"

        if final_pred == "Assemble3" and last_conf >= CONF_THRESHOLD:
            detect_count["Assemble3"] += 1
        else:
            detect_count["Assemble3"] = 0

        if detect_count["Assemble3"] >= DETECT_NEEDED:
            step_started["Assemble3"] = True
            if current_alert_text == "ROBOT PREPARING PRODUCT (ASSEMBLE 3)":
                current_alert_text = None

        # ==============================================
        # TOP-LEFT OVERLAY (顯示狀態機校正結果)
        # ==============================================
        cv2.putText(frame, f"Infer FPS: {last_infer_fps:.2f}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        if raw_pred != final_pred:
            if final_pred == "Waiting" and raw_pred == "Assemble3":
                cv2.putText(frame, f"Pred: {final_pred} (Cooldown {COOLDOWN_A2_TO_A3}s)", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"Pred: {final_pred} (Fixed from {raw_pred})", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"Pred: {final_pred} ({last_conf:.2f})", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # ==============================================
        # CENTER ALERT OVERLAY
        # ==============================================
        if current_alert_text is not None and int(time.time() * 2) % 2 == 0:
            draw_center_alert_transparent(
                frame,
                current_alert_text,
                bg_color=(0, 0, 255),
                text_color=(255, 255, 255),
                font_scale=1.4,
                thickness=3,
                pad_x=40,
                pad_y=24,
                alpha=0.78
            )

        # ==============================================
        # LEFT PANEL
        # ==============================================
        y_left = 110
        draw_left_text(frame, "PENDING ACTIVITIES", y_left, scale=0.75, color=(0, 255, 255), thickness=2)
        y_left += 35

        for name in display_classes:
            if not class_states[name]["is_ok"]:
                draw_left_text(frame, f"{name}: NOT OK", y_left, scale=0.7, color=(0, 0, 255), thickness=2)
                y_left += 28

        # ==============================================
        # RIGHT PANEL
        # ==============================================
        y_right = 110
        draw_right_text(frame, "COMPLETED", y_right, scale=0.75, color=(0, 255, 0), thickness=2)
        y_right += 35

        for name in display_classes:
            if class_states[name]["is_ok"]:
                draw_right_text(frame, f"{name}: DONE", y_right, scale=0.7, color=(0, 255, 0), thickness=2)
                y_right += 28

        display = cv2.resize(frame, (1820, 1080), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Action Validation", display)

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
    model_path = "MoreAngle_EDA.pth"

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
        camera_index=0,
        model=model,
        transform=transform,
        class_names=class_names,
        clip_len=16,
        device=device,
    )


if __name__ == "__main__":
    main()