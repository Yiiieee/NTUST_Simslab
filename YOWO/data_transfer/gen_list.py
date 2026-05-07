import os

# 設定你的資料路徑
BASE_PATH = "/home/simslab/Desktop/api/Ziv/YOWO/pipeline_data/rgb/train"
OUTPUT_FILE = "/home/simslab/Desktop/api/Ziv/YOWO/pipeline_data/trainlist.txt"

video_folders = sorted(os.listdir(BASE_PATH))
all_frames = []

for folder in video_folders:
    folder_path = os.path.join(BASE_PATH, folder)
    if os.path.isdir(folder_path):
        # 抓取該資料夾下所有的 .jpg 檔案
        frames = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
        for f in frames:
            # 取得檔名（不含副檔名），例如 0430_1/00001
            frame_name = os.path.splitext(f)[0]
            all_frames.append(f"{folder}/{frame_name}")

with open(OUTPUT_FILE, 'w') as f:
    for line in all_frames:
        f.write(line + '\n')

print(f"成功！已生成 {len(all_frames)} 筆影格資料列於 trainlist.txt")