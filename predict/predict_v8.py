<<<<<<< HEAD
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
import time
#import collectionsPYT
import collections
from collections import deque, Counter
from typing import Optional, Tuple, Dict


# =========================================================
# SAFE MODEL LOADER (ENV-PROOF)
# =========================================================
def _get_r3d_transform_from_weights(weights=None):
    # Defaults for R3D Kinetics
    mean = [0.43216, 0.394666, 0.37645]
    std  = [0.22803, 0.22145, 0.216989]

    if weights is not None:
        try:
            meta = weights.meta
            if "mean" in meta and "std" in meta:
                mean = meta["mean"]
                std = meta["std"]
        except Exception:
            pass

    transform = transforms.Compose([
        transforms.Resize((128, 171)), #
        # transforms.Resize((1280, 171)),
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform


def _safe_load_state_dict(model: nn.Module, ckpt_path: str, num_classes: int):
    state = torch.load(ckpt_path, map_location="cpu")

    # If checkpoint is a dict (common), try common keys
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    # Hard check: fc weight shape must match num_classes
    # Avoid silent strict=False that makes fc random.
    if "fc.weight" in state:
        w = state["fc.weight"]
        if hasattr(w, "shape") and w.shape[0] != num_classes:
            raise RuntimeError(
                f"Checkpoint fc.weight shape {tuple(w.shape)} != expected ({num_classes}, ...).\n"
                f"Your class_names order/count is NOT matching training.\n"
                f"Fix class_names to exactly match training."
            )

    missing, unexpected = model.load_state_dict(state, strict=False)

    # If fc.weight not loaded (missing), warn hard
    # This usually means mismatch architecture or checkpoint.
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
            weights = R3D_18_Weights.DEFAULT
            model = r3d_18(weights=weights)
            transform = _get_r3d_transform_from_weights(weights)
        except Exception:
            # weights might fail (no internet / older torchvision)
            model = r3d_18(weights=None)
            transform = _get_r3d_transform_from_weights(None)

        model.fc = nn.Linear(model.fc.in_features, num_classes)

        _safe_load_state_dict(model, model_path, num_classes)
        model.eval()
        return model, transform

    except Exception as e_r3d:
        print(f"[WARN] R3D load failed: {e_r3d}")

    # ---------- Fallback: ResNet18 adapted to 3D (VERY RISKY) ----------
    # Only do this if checkpoint keys look compatible.
    try:
        from torchvision.models import resnet18, ResNet18_Weights

        # Build model
        try:
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
        except Exception:
            model = resnet18(weights=None)

        # Adapt conv1 to 3D input (not truly equivalent to R3D)
        model.conv1 = nn.Conv3d(
            3, 64, kernel_size=(3, 7, 7),
            stride=(1, 2, 2), padding=(1, 3, 3), bias=False
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Transform for ResNet-style
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Try load checkpoint safely (will hard stop if fc mismatch)
        _safe_load_state_dict(model, model_path, num_classes)
        model.eval()
        print("[WARN] Using ResNet18-3D fallback. Accuracy may be unreliable vs training.")
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
    x = margin
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# =========================================================
# LIVE PREDICTION + TEMPORAL VALIDATION + DUAL-COLUMN UI
# =========================================================
def predict_live_camera(camera_index, model, transform, class_names, clip_len=16, device="cpu"):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Cannot open video/camera:", camera_index)
        
        return
    
    win_name = "Action Validation"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps is None or input_fps <= 1:
        input_fps = 30.0  # safe fallback

    clip_duration = clip_len / input_fps
    min_stay_time = 3.0
    conf_threshold = 0.6

    # Hide classes from OK/NOT OK panels
    hidden_classes = {"Take", "Screwin"}  # edit here
    display_classes = [c for c in class_names if c not in hidden_classes]
    class_states = {name: {"active_time": 0.0, "is_ok": False} for name in display_classes}

    # Save video
    save_video = True
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_path = "output_result_envproof3.avi"
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

    while True: 
        ret, frame = cap.read()
        if not ret:
            break

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

            # Temporal validation ONLY for display classes
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

        # TOP-LEFT overlayF
        cv2.putText(frame, f"Infer FPS: {last_infer_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Pred: {last_pred_class} ({last_conf:.2f})", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        # LEFT panel
        y_left = 110
        draw_left_text(frame, "PENDING ACTIVITIES", y_left, scale=0.55, color=(0, 255, 255), thickness=1)
        y_left += 35
        for name in display_classes:
            if not class_states[name]["is_ok"]:
                draw_left_text(frame, f"{name}: n/a", y_left, scale=0.5, color=(0, 0, 255), thickness=1)
                y_left += 28

        # RIGHT panel
        y_right = 110
        draw_right_text(frame, "COMPLETED", y_right, scale=0.55, color=(0, 255, 0), thickness=1)
        y_right += 35
        for name in display_classes:
            if class_states[name]["is_ok"]:
                draw_right_text(frame, f"{name}: done", y_right, scale=0.5, color=(0, 255, 0), thickness=1)
                y_right += 28

        # cv2.imshow("Action Validation", frame)
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
=======
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
import time
#import collectionsPYT
import collections
from collections import deque, Counter
from typing import Optional, Tuple, Dict


# =========================================================
# SAFE MODEL LOADER (ENV-PROOF)
# =========================================================
def _get_r3d_transform_from_weights(weights=None):
    # Defaults for R3D Kinetics
    mean = [0.43216, 0.394666, 0.37645]
    std  = [0.22803, 0.22145, 0.216989]

    if weights is not None:
        try:
            meta = weights.meta
            if "mean" in meta and "std" in meta:
                mean = meta["mean"]
                std = meta["std"]
        except Exception:
            pass

    transform = transforms.Compose([
        transforms.Resize((128, 171)), #
        # transforms.Resize((1280, 171)),
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform


def _safe_load_state_dict(model: nn.Module, ckpt_path: str, num_classes: int):
    state = torch.load(ckpt_path, map_location="cpu")

    # If checkpoint is a dict (common), try common keys
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    # Hard check: fc weight shape must match num_classes
    # Avoid silent strict=False that makes fc random.
    if "fc.weight" in state:
        w = state["fc.weight"]
        if hasattr(w, "shape") and w.shape[0] != num_classes:
            raise RuntimeError(
                f"Checkpoint fc.weight shape {tuple(w.shape)} != expected ({num_classes}, ...).\n"
                f"Your class_names order/count is NOT matching training.\n"
                f"Fix class_names to exactly match training."
            )

    missing, unexpected = model.load_state_dict(state, strict=False)

    # If fc.weight not loaded (missing), warn hard
    # This usually means mismatch architecture or checkpoint.
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
            weights = R3D_18_Weights.DEFAULT
            model = r3d_18(weights=weights)
            transform = _get_r3d_transform_from_weights(weights)
        except Exception:
            # weights might fail (no internet / older torchvision)
            model = r3d_18(weights=None)
            transform = _get_r3d_transform_from_weights(None)

        model.fc = nn.Linear(model.fc.in_features, num_classes)

        _safe_load_state_dict(model, model_path, num_classes)
        model.eval()
        return model, transform

    except Exception as e_r3d:
        print(f"[WARN] R3D load failed: {e_r3d}")

    # ---------- Fallback: ResNet18 adapted to 3D (VERY RISKY) ----------
    # Only do this if checkpoint keys look compatible.
    try:
        from torchvision.models import resnet18, ResNet18_Weights

        # Build model
        try:
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
        except Exception:
            model = resnet18(weights=None)

        # Adapt conv1 to 3D input (not truly equivalent to R3D)
        model.conv1 = nn.Conv3d(
            3, 64, kernel_size=(3, 7, 7),
            stride=(1, 2, 2), padding=(1, 3, 3), bias=False
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Transform for ResNet-style
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Try load checkpoint safely (will hard stop if fc mismatch)
        _safe_load_state_dict(model, model_path, num_classes)
        model.eval()
        print("[WARN] Using ResNet18-3D fallback. Accuracy may be unreliable vs training.")
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
    x = margin
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# =========================================================
# LIVE PREDICTION + TEMPORAL VALIDATION + DUAL-COLUMN UI
# =========================================================
def predict_live_camera(camera_index, model, transform, class_names, clip_len=16, device="cpu"):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Cannot open video/camera:", camera_index)
        
        return
    
    win_name = "Action Validation"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps is None or input_fps <= 1:
        input_fps = 30.0  # safe fallback

    clip_duration = clip_len / input_fps
    min_stay_time = 3.0
    conf_threshold = 0.6

    # Hide classes from OK/NOT OK panels
    hidden_classes = {"Take", "Screwin"}  # edit here
    display_classes = [c for c in class_names if c not in hidden_classes]
    class_states = {name: {"active_time": 0.0, "is_ok": False} for name in display_classes}

    # Save video
    save_video = True
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_path = "output_result_envproof3.avi"
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

    while True: 
        ret, frame = cap.read()
        if not ret:
            break

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

            # Temporal validation ONLY for display classes
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

        # TOP-LEFT overlayF
        cv2.putText(frame, f"Infer FPS: {last_infer_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Pred: {last_pred_class} ({last_conf:.2f})", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        # LEFT panel
        y_left = 110
        draw_left_text(frame, "PENDING ACTIVITIES", y_left, scale=0.55, color=(0, 255, 255), thickness=1)
        y_left += 35
        for name in display_classes:
            if not class_states[name]["is_ok"]:
                draw_left_text(frame, f"{name}: n/a", y_left, scale=0.5, color=(0, 0, 255), thickness=1)
                y_left += 28

        # RIGHT panel
        y_right = 110
        draw_right_text(frame, "COMPLETED", y_right, scale=0.55, color=(0, 255, 0), thickness=1)
        y_right += 35
        for name in display_classes:
            if class_states[name]["is_ok"]:
                draw_right_text(frame, f"{name}: done", y_right, scale=0.5, color=(0, 255, 0), thickness=1)
                y_right += 28

        # cv2.imshow("Action Validation", frame)
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
>>>>>>> 2a6bd7f1f941ef825654fcd4c38b276dd00248b6
