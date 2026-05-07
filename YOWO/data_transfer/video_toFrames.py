import cv2
import os
import glob

# --- 設定路徑 ---
# 原始影片所在資料夾
INPUT_VIDEO_DIR = 'data_video' 
# 輸出的圖片路徑，建議直接對應 YOWO 要求的結構
OUTPUT_RGB_DIR = 'rgb/train' 

def process_all_videos():
    # 建立輸出根目錄
    if not os.path.exists(OUTPUT_RGB_DIR):
        os.makedirs(OUTPUT_RGB_DIR)
        print(f"建立目錄: {OUTPUT_RGB_DIR}")

    # 抓取 data_video 下所有的 mp4 檔案
    video_files = glob.glob(os.path.join(INPUT_VIDEO_DIR, "*.mp4"))
    
    if not video_files:
        print(f"錯誤：在 {INPUT_VIDEO_DIR} 中找不到任何 .mp4 檔案。")
        return

    print(f"找到 {len(video_files)} 部影片，準備開始轉換...")

    for video_path in video_files:
        # 取得影片檔案名稱（去副檔名），作為子資料夾名稱
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_folder = os.path.join(OUTPUT_RGB_DIR, video_name)
        
        if not os.path.exists(video_output_folder):
            os.makedirs(video_output_folder)

        print(f"正在處理影片: {video_name} ...", end="\r")

        cap = cv2.VideoCapture(video_path)
        frame_count = 1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOWO 慣例：檔名補零成 5 位數，如 00001.jpg
            frame_filename = f"{frame_count:05d}.jpg"
            save_path = os.path.join(video_output_folder, frame_filename)
            
            # 儲存圖片
            cv2.imwrite(save_path, frame)
            frame_count += 1
            
        cap.release()
        print(f"影片 {video_name} 轉換完成，共 {frame_count-1} 張圖片。")

if __name__ == "__main__":
    process_all_videos()
    print("\n所有影片轉換流程已結束！")