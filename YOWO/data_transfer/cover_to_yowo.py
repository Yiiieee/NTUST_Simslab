import pandas as pd
import os
import glob

# 1. 動作對照表 (統一用小寫，腳本會自動轉換比對)
ACTION_MAP = {
    'waiting': 0,
    'brush': 1,
    'screwin': 2,
    'assemble1': 3,
    'assemble2': 4,
    'assemble3': 5,
    'take': 6
}

def process_labels():
    # 搜尋 data_csv 資料夾下所有的 csv 檔案
    csv_files = glob.glob('data_csv/*.csv')
    
    if not csv_files:
        print("❌ 錯誤：在 data_csv 中找不到任何 .csv 檔案。")
        return

    for csv_path in csv_files:
        video_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_dir = f'annotations/train/{video_name}'
        
        # 解決 Excel 產生的 CSV 編碼問題
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='cp950')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"📊 正在處理影片: {video_name}", end=" ")

        count = 0
        for i, row in df.iterrows():
            try:
                # 根據截圖：
                # 第 2 欄 (Index 2) 是 Frame
                # 第 3 欄 (Index 3) 是 標記行為
                frame_val = row.iloc[2] 
                action_val = row.iloc[3]
                
                if pd.isna(frame_val) or pd.isna(action_val):
                    continue
                
                # 關鍵處理：轉成字串 -> 去掉空格 -> 轉成小寫
                action = str(action_val).strip().lower()
                frame_id = int(frame_val)
                
                if action in ACTION_MAP:
                    class_id = ACTION_MAP[action]
                    # 生成 YOWOv2 格式：00001.txt
                    txt_path = os.path.join(output_dir, f'{frame_id:05d}.txt')
                    with open(txt_path, 'w') as f:
                        # 格式：[類別ID] [x1 y1 x2 y2] (暫設全畫面 0 0 1 1)
                        f.write(f"{class_id} 0.0 0.0 1.0 1.0\n")
                    count += 1
            except Exception:
                continue
        
        if count == 0:
            # 診斷資訊：如果還是 0，印出抓到的第一筆資料內容
            print(f" ⚠️ 警告：匹配失敗。請檢查 CSV。第一筆偵測到的行為文字為: '{df.iloc[0].iloc[3]}'")
        else:
            print(f" ✅ 成功生成 {count} 個標註檔。")

if __name__ == "__main__":
    process_labels()
    print("\n🚀 標籤轉換流程已全部完成！")