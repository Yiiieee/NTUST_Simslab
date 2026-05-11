import cv2

def list_ports(max_tested=10):
    """
    測試從 0 到 max_tested 的相機連接埠，並回傳可用的 Index 列表。
    """
    available_ports = []
    print(f"🔍 正在掃描前 {max_tested} 個相機索引...\n" + "-"*40)
    
    for i in range(max_tested):
        # 嘗試連接相機，cv2.CAP_V4L2 可以在 Linux (Ubuntu) 下加速開啟速度並減少報錯
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        
        if cap.isOpened():
            # 嘗試讀取一張畫面，用來驗證這是否為真正的影像流 (過濾掉 metadata 節點)
            is_reading, img = cap.read()
            
            if is_reading:
                print(f"✅ 找到相機 Index: [{i}] - 可正常讀取畫面")
                available_ports.append(i)
            else:
                print(f"⚠️ 找到相機 Index: [{i}] - 但無法讀取畫面 (可能是中介節點或被其他程式佔用)")
            
            # 測試完畢後務必釋放資源
            cap.release()
            
    print("-" * 40)
    if not available_ports:
        print("❌ 沒有找到任何可用的相機。")
    else:
        print(f"🎯 總結：你可以在 OpenCV 中使用的相機 Index 為 {available_ports}")

if __name__ == "__main__":
    list_ports()