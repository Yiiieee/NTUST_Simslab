import os
import numpy as np
import torch
import torch.utils.data as data
import cv2
from PIL import Image  # 1. 引入 PIL 以支援 transforms.py

# --- 第一個類別：訓練用 (Frame-level) ---
class UCF_JHMDB_Dataset(data.Dataset):
    def __init__(self, 
                 data_root, 
                 dataset='ucf24', 
                 img_size=224,
                 len_clip=16, 
                 is_train=False, 
                 transform=None, 
                 sampling_rate=1):
        self.data_root = data_root
        self.dataset = dataset
        self.img_size = img_size
        self.len_clip = len_clip
        self.is_train = is_train
        self.transform = transform
        self.sampling_rate = sampling_rate

        if self.is_train:
            self.split_list = 'trainlist.txt'
        else:
            self.split_list = 'testlist.txt'

        self.dataset_list = []
        list_path = os.path.join(data_root, self.split_list)
        
        if not os.path.exists(list_path):
            print(f"⚠️ 警告: 找不到 {list_path}，嘗試讀取 trainlist.txt。")
            list_path = os.path.join(data_root, 'trainlist.txt')

        with open(list_path, 'r') as f:
            for line in f:
                self.dataset_list.append(line.strip())

        self.num_samples = len(self.dataset_list)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.pull_item(index)

    def pull_item(self, index):
        frame_path_info = self.dataset_list[index]
        img_file = os.path.join(self.data_root, 'rgb', 'train', frame_path_info + '.jpg')
        annot_file = os.path.join(self.data_root, 'annotations', 'train', frame_path_info + '.txt')

        filename = os.path.basename(img_file)
        img_id_str = os.path.splitext(filename)[0]
        try:
            img_id = int(img_id_str)
        except ValueError:
            img_id = 0

        video_clip = []
        for i in range(self.len_clip):
            back_idx = img_id - i * self.sampling_rate
            if back_idx < 1:
                back_idx = 1
            
            dir_name = os.path.dirname(frame_path_info)
            clip_frame_name = f"{back_idx:05d}.jpg"
            clip_path = os.path.join(self.data_root, 'rgb', 'train', dir_name, clip_frame_name)
            
            frame = cv2.imread(clip_path)
            if frame is None:
                frame = cv2.imread(img_file)
            
            # 轉換顏色並轉為 PIL Image 格式，解決 .crop 和 .height 報錯
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame) 
            video_clip.append(frame)

        video_clip.reverse()

        target = []
        if os.path.exists(annot_file):
            with open(annot_file, 'r') as f:
                for line in f:
                    line = line.strip().split(' ')
                    if len(line) == 5:
                        cls_id = int(line[0])
                        x1, y1, x2, y2 = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                        target.append([x1, y1, x2, y2, cls_id])
        
        if len(target) == 0:
            target = np.zeros((0, 5))
        else:
            target = np.array(target)

        # 此時 video_clip 裡面是 PIL 圖片，符合 transforms.py 的預期
        if self.transform is not None:
            video_clip, target = self.transform(video_clip, target)

        return img_id, video_clip, target


# --- 第二個類別：評估用 (Video-level) ---
class UCF_JHMDB_VIDEO_Dataset(data.Dataset):
    def __init__(self, 
                 data_root, 
                 dataset='ucf24', 
                 img_size=224,
                 len_clip=16, 
                 transform=None, 
                 sampling_rate=1):
        self.data_root = data_root
        self.dataset = dataset
        self.img_size = img_size
        self.len_clip = len_clip
        self.transform = transform
        self.sampling_rate = sampling_rate
        self.video_list = sorted(os.listdir(os.path.join(data_root, 'rgb/train')))

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        return None, None, None