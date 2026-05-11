import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import time
import collections 

def load_model(model_path, num_classes):
    model = None
    transform = None
    pretained_weights_loaded = False

    try:
        from torchvision.models.video import r3d_18, R3D_18_Weights
        try:
            weights = R3D_18_Weights.DEFAULT
            model = r3d_18(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            # print("Loaded R3D model with default weights.")
            pretained_weights_loaded = True
            resize_size = (128, 171)
            crop_size = (112, 112)
            try:
                 mean = weights.meta['mean']
                 std = weights.meta['std']
            except (AttributeError, KeyError):
                #  print("Could not get mean/std from R3D weights meta, using defaults.")
                 mean=[0.43216, 0.394666, 0.37645]
                 std=[0.22803, 0.22145, 0.216989]

            transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        except Exception as e:
            print(f"Failed to load R3D with default weights: {e}")
            model = None
            transform = None

        if model is None:
            print("Attempting to load R3D model structure without pretrained weights.")
            model = r3d_18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            resize_size = (128, 171); crop_size = (112, 112)
            mean=[0.43216, 0.394666, 0.37645]; std=[0.22803, 0.22145, 0.216989]
            transform = transforms.Compose([
                transforms.Resize(resize_size), 
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(), 
                transforms.Normalize(mean=mean, std=std)
            ])

    except (ImportError, AttributeError) as e:
        print(f"R3D model import failed ({e}). Falling back to ResNet18-based 3D Conv.")
        model = None

    if model is None:
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            try:
                weights = ResNet18_Weights.DEFAULT
                model = resnet18(weights=weights)
                pretained_weights_loaded = True
                resize_size = (256, 256); crop_size = (224, 224)
                try:
                    mean = weights.meta['mean']
                    std = weights.meta['std']
                except (AttributeError, KeyError):
                    #  print("Could not get mean/std from ResNet weights meta, using defaults.")
                     mean=[0.485, 0.456, 0.406]
                     std=[0.229, 0.224, 0.225]
                transform = transforms.Compose([
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])

            except Exception as e:
                print(f"Failed to load ResNet18 with default weights: {e}")
                model = resnet18(weights=None)
                print("Loaded ResNet18 structure without pretrained weights.")
                resize_size = (256, 256); crop_size = (224, 224)
                mean=[0.485, 0.456, 0.406]; std=[0.229, 0.224, 0.225]
                transform = transforms.Compose([
                    transforms.Resize(resize_size), transforms.CenterCrop(crop_size),
                    transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)
                ])

            # print("Adapting ResNet18 for 3D input...")
            model.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        except ImportError as e:
             raise ImportError(f"Failed to import ResNet18. Error: {e}")

    if model is None: 
        raise RuntimeError("Failed to initialize any model architecture.")

    try:
        # print(f"Loading fine-tuned weights from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    except Exception as e:
        print(f"Error loading state dict from {model_path}: {e}")
        raise

    model.eval()

    if transform is None: raise RuntimeError("Transform pipeline was not created.")
    # print("Model and transform ready.")
    return model, transform

def predict_live_camera(camera_index, model, transform, class_names, clip_len=16, device='cpu'):
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(f"Camera {camera_index} opened. Resolution: {frame_width}x{frame_height}")

    frame_buffer = collections.deque(maxlen=clip_len)

    model = model.to(device)
    model.eval() 

    last_prediction = "Collecting frames..." 
    last_fps = 0.0 

    print("Starting prediction... Press 'q' to quit.")

    while True:
        ret, frame = cap.read() 

        if not ret:
            print("Error: Failed to grab frame or camera disconnected.")
            break

        frame_buffer.append(frame.copy())

        predicted_class = last_prediction
        current_fps = last_fps

        if len(frame_buffer) == clip_len:
            start_time = time.time() 

            clip_bgr = list(frame_buffer)

            clip_tensor_list = []
            try:
                for frame_bgr_in_clip in clip_bgr:
                    frame_rgb = cv2.cvtColor(frame_bgr_in_clip, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    transformed_frame = transform(pil_img)
                    clip_tensor_list.append(transformed_frame)

                input_tensor = torch.stack(clip_tensor_list, dim=1).unsqueeze(0).to(device)

                with torch.no_grad(): 
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1) 
                    confidence, pred = torch.max(probabilities, 1)

                predicted_class = class_names[pred.item()] 
                confidence_score = confidence.item() 
                end_time = time.time() # 結束計時
                processing_time = end_time - start_time
                current_fps = 1.0 / processing_time if processing_time > 0 else float('inf')

                last_prediction = f"{predicted_class} ({confidence_score:.2f})" # Show confidence
                last_fps = current_fps


            except Exception as e:
                 print(f"\nError during prediction or processing clip: {e}")
                 predicted_class = "Prediction Error"
                 current_fps = 0.0
                 # frame_buffer.clear()


        display_frame = frame 
        text_fps = f"FPS: {current_fps:.2f}"
        text_class = f"Class: {predicted_class}" 

        # cv2.putText(display_frame, text_fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #             1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display_frame, text_class, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow(' Press Q to Quit', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows() 
    print("Quit.")

def main():
    # 設定參數
    model_path = r'/home/simslab/Desktop/api/run1.pth'  # 模型路徑
    class_names = ['Assemble', 'Affix adhesive', 'Scan stickers', 'Tighten screws', 'Affix stickers', 'Waiting']
    clip_length = 16  
    camera_index = 0  # 攝影機
    # camera_index = r"/home/simslab/Desktop/normal_m1_new.mp4"  # 測試影片路徑

    
    
              
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    num_classes = len(class_names)
    model, transform = load_model(model_path, num_classes)

    predict_live_camera(
        camera_index=camera_index,
        model=model,
        transform=transform,
        class_names=class_names,
        clip_len=clip_length,
        device=device 
    )


    # print("Live prediction stopped.")

if __name__ == '__main__':
    main()

    