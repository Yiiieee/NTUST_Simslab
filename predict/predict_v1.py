import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
from tqdm import tqdm
import time
import collections

def load_model(model_path, num_classes):
    model = None
    transform = None
    pretained_weights_loaded = False

    try:
        from torchvision.models.video import r3d_18, R3D_18_Weights
        print("Attempting to load R3D model...")
        try:
            weights = R3D_18_Weights.DEFAULT
            model = r3d_18(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            print("Loaded R3D model with default weights.")
            pretained_weights_loaded = True
            resize_size = (128, 171)
            crop_size = (112, 112)
            try:
                 mean = weights.meta['mean']
                 std = weights.meta['std']
            except (AttributeError, KeyError):
                 print("Could not get mean/std from R3D weights meta, using defaults.")
                 mean=[0.43216, 0.394666, 0.37645]
                 std=[0.22803, 0.22145, 0.216989]

            transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(), 
                transforms.Normalize(mean=mean, std=std)
            ])
            # print("Created frame-by-frame transform for R3D.")

        except Exception as e: 
            print(f"Failed to load R3D with default weights: {e}")
            model = None 
            transform = None

        if model is None:
            print("Attempting to load R3D model structure without pretrained weights.")
            model = r3d_18(weights=None) # Load architecture only
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            resize_size = (128, 171)
            crop_size = (112, 112)
            mean=[0.43216, 0.394666, 0.37645] # Still use common defaults
            std=[0.22803, 0.22145, 0.216989]
            transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            # print("Loaded R3D structure, created default frame-by-frame transform.")


    except (ImportError, AttributeError) as e:
        print(f"R3D model import failed ({e}). Falling back to ResNet18-based 3D Conv.")
        model = None 

    if model is None:
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            print("Attempting to load ResNet18 base for 3D Conv...")
            try:
                weights = ResNet18_Weights.DEFAULT
                model = resnet18(weights=weights) # Load pretrained 2D ResNet
                print("Loaded ResNet18 with default weights.")
                pretained_weights_loaded = True
                resize_size = (256, 256) 
                crop_size = (224, 224)  
                try:
                    mean = weights.meta['mean']
                    std = weights.meta['std']
                except (AttributeError, KeyError):
                     print("Could not get mean/std from ResNet weights meta, using defaults.")
                     mean=[0.485, 0.456, 0.406]
                     std=[0.229, 0.224, 0.225]

                transform = transforms.Compose([
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
                print("Created frame-by-frame transform for ResNet base.")

            except Exception as e:
                print(f"Failed to load ResNet18 with default weights: {e}")
                model = resnet18(weights=None) # Load architecture only
                print("Loaded ResNet18 structure without pretrained weights.")
                # Use default ResNet transforms
                resize_size = (256, 256)
                crop_size = (224, 224)
                mean=[0.485, 0.456, 0.406]
                std=[0.229, 0.224, 0.225]
                transform = transforms.Compose([
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
                # print("Created default frame-by-frame transform for ResNet base.")

            # print("Adapting ResNet18 for 3D input...")
            model.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            # print("ResNet18 adapted for 3D.")

        except ImportError as e:
             raise ImportError(f"Failed to import ResNet18. Please ensure torchvision is installed correctly. Error: {e}")


    if model is None:
        raise RuntimeError("Failed to initialize any model architecture.")

    try:
        print(f"Loading fine-tuned weights from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    except Exception as e:
        print(f"Error loading state dict from {model_path}: {e}")
        raise

    model.eval()

    if transform is None:
         raise RuntimeError("Transform pipeline was not created.")

    print("Model and transform ready.")
    return model, transform

def predict_video_and_save(video_path, model, transform, class_names, clip_len=16, output_video_path="output_video.mp4"):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # 獲取影片屬性
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path}")
    print(f"Original FPS: {fps:.2f}")
    print(f"Total frames: {frame_count}")
    print(f"Resolution: {frame_width}x{frame_height}")

    original_frames = []
    print("Reading frames...")
    with tqdm(total=frame_count, desc="Reading frames") as pbar:
        while True:
            ret, frame = cap.read() 
            if not ret:
                break
            original_frames.append(frame)
            pbar.update(1)
    cap.release()

    if not original_frames:
        print("Error: No frames were read from the video.")
        return

    predictions = []
    processed_frames_for_video = [] # 儲存帶有文字的 BGR 影格

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    with torch.no_grad():
        for i in tqdm(range(len(original_frames)), desc="Predicting"):
            start_time = time.time() # 開始計時

            start_idx = max(0, i - clip_len + 1)
            clip_bgr = original_frames[start_idx:i+1]

            if len(clip_bgr) < clip_len:
                padding = clip_len - len(clip_bgr)
                clip_bgr = [clip_bgr[0]] * padding + clip_bgr

            clip_tensor_list = []
            for frame_bgr in clip_bgr:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                transformed_frame = transform(pil_img)
                clip_tensor_list.append(transformed_frame)

            try:
                input_tensor = torch.stack(clip_tensor_list, dim=1).unsqueeze(0).to(device) # Shape: [1, C, clip_len, H, W]
            except RuntimeError as e:
                 print(f"\nError stacking tensors at frame {i}. Clip length: {len(clip_tensor_list)}. Expected length: {clip_len}.")
                 print(f"Individual tensor shape: {clip_tensor_list[0].shape if clip_tensor_list else 'N/A'}")
                 # Attempt to pad if tensor list is not empty but maybe wrong shape?
                 if clip_tensor_list and clip_tensor_list[0].shape[1:] != (transform.transforms[-2].size if hasattr(transform.transforms[-2], 'size') else (112, 112)): # Assuming CenterCrop size check
                     print("Tensor shape mismatch after transform.")
                 # Simple skip strategy: duplicate last known good prediction or use 'Waiting'
                 if predictions:
                     predicted_class = predictions[-1]
                 else:
                     predicted_class = "Waiting" # Default if error on first frame
                 predictions.append(predicted_class)
                 # Use original frame without prediction text for output video? Or add error text?
                 frame_with_text = original_frames[i].copy()
                 cv2.putText(frame_with_text, f"Prediction Error", (10, 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                 processed_frames_for_video.append(frame_with_text)
                 continue # Skip to next frame


            # --- 模型預測 ---
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            predicted_class = class_names[pred.item()]
            predictions.append(predicted_class) # 儲存預測結果

            end_time = time.time() # 結束計時
            processing_time = end_time - start_time
            current_fps = 1.0 / processing_time if processing_time > 0 else float('inf')

            # --- 在 *原始 BGR* 影格上添加文字 ---
            frame_to_write = original_frames[i].copy() # 複製當前的 BGR 影格
            text_fps = f"FPS: {current_fps:.2f}"
            text_class = f"Class: {predicted_class}"
            # 添加 FPS 文字 (左上角)
            cv2.putText(frame_to_write, text_fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            # 添加類別文字 (FPS 下方)
            cv2.putText(frame_to_write, text_class, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            # 將處理過的 BGR 影格添加到列表中，用於寫入影片
            processed_frames_for_video.append(frame_to_write)

    # --- 寫入輸出影片 ---
    if processed_frames_for_video:
        print(f"Writing output video to {output_video_path}...")
        # 定義影片編碼器和創建 VideoWriter 物件
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 或者 'XVID', 'MJPG' 等
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # XVID 通常兼容性更好
        # 使用原始影片的 FPS 和解析度
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        for frame in tqdm(processed_frames_for_video, desc="Writing video"):
            out.write(frame) # 寫入 BGR 影格

        out.release() # 釋放 VideoWriter
        print(f"Output video saved successfully: {output_video_path}")
    else:
        print("No frames were processed to write to video.")

    
    # ########
    # cap.release()
    # cv2.destroyAllWindows() 
    # print("Quit.")

# 主函數
def main():
    # 設定參數
    model_path = r'/home/simslab/Desktop/api/run1.pth'  # 模型路徑
    # video_path = r"/home/simslab/Desktop/normal_m1_new.mp4"  # 測試視頻路徑
    video_path = 0  # 使用攝影機
    # output_video_path = "run_2_output_video.avi" # 輸出影片路徑 (使用 .avi 搭配 XVID)
    class_names = ['Assemble', 'Brush', 'Scan', 'Screwin', 'Sticker', 'Waiting']  # 替換為您的類別
    clip_length = 16  # 每個視頻採樣的幀數 (與模型訓練時一致)

    # 檢查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加載模型和對應的轉換
    num_classes = len(class_names)
    model, transform = load_model(model_path, num_classes) # 獲取模型和轉換

    # 預測視頻並保存結果影片
    predict_video_and_save(
        video_path=video_path,
        model=model,
        transform=transform, # 傳遞獲取到的轉換
        class_names=class_names,
        clip_len=clip_length,
        # output_video_path=output_video_path # 傳遞輸出影片路徑
    ) 

    # print("Processing completed!")

if __name__ == '__main__':
    main()