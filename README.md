<div align="center">
  <h1>🤖 NTUST 人機協作系統 (Human-Machine Collaboration)</h1>
  <p><strong>結合即時動作辨識與物料追蹤的智慧組裝線系統</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C" alt="PyTorch">
    <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8" alt="OpenCV">
    <img src="https://img.shields.io/badge/Supabase-Database-3ECF8E" alt="Supabase">
  </p>
</div>

---

## 📖 Overview

本專案為國立臺灣科技大學 (NTUST) 所開發的**人機協作 (HMC) 系統**。本系統旨在透過整合**即時 3D 人體動作辨識**與**自動化物料計算及庫存管理**，來提升組裝線的作業效率與品質。

藉由即時監控操作員的組裝動作，並結合嚴謹的狀態機 (State Machine) 進行動作順序驗證，系統能有效確保組裝流程的正確性，同時在動態儀表板上即時更新物料庫存數據。

## ✨ Key Features

- 🏃‍♂️ **Real-Time Action Recognition**：採用 3D ResNet-18 模型，即時處理連續影像特徵，精準分類人體組裝動作（如：`Assemble1`, `Assemble2`, `Assemble3`, `Brush`, `Screwin`, `Take`, `Waiting`）。
- 🛡️ **State Machine Validation**：利用嚴格的動作相依性、最短持續時間與步驟間的冷卻時間設定，智慧過濾模型預測雜訊，杜絕錯誤的組裝順序。
- 📦 **Material Counting**：整合掃描或計算機制來追蹤組裝過程中消耗的物料。
- 📊 **Live Inventory Dashboard**：使用 `Tkinter` 打造視覺化 GUI，與 **Supabase** 雲端資料庫即時同步。當物料庫存降至 `0` 時，畫面會出現醒目的閃爍警示動畫。
- 🎥 **Multi-Camera Sync**：提供 `open2cam.py` 工具，支援雙攝影機同步錄影，方便進行多視角資料集的收集。

## 📂 Repository Structure

```text
NTUST_Human-machine-collaboration/
├── 3D_ResNet/       # 3D ResNet 模型架構、訓練與驗證腳本
├── predict/         # 即時推論程式 (如 predict_v11.py)，包含 UI 繪製與狀態機邏輯
├── counting/        # 物料計算系統、Supabase 串接與即時儀表板 (dashboard4.py)
├── open2cam.py      # 雙鏡頭同步錄影的實用工具
└── README.md        # 專案說明文件
```



## 🚀 Getting Started

### 1. 環境建置

建議使用 Conda 或虛擬環境來安裝依賴套件：
```bash
conda env create -f counting/environment.yml
conda activate <env_name>
```
*(請確認安裝與您的 CUDA 版本相符的 PyTorch)。*

### 2. 啟動即時動作辨識

執行以下指令，透過網路攝影機開始即時動作辨識與狀態機驗證：
```bash
python predict/predict_v11.py
```
*(請確認已將訓練好的權重檔 `MoreAngle_EDA.pth` 放置於正確路徑，或於腳本中自行修改)。*

### 3. 啟動物料計算與儀表板

若要同時啟動物料計算服務與即時庫存儀表板，請執行：
```bash
python counting/Start_system.py
```
這將在背景啟動物料計算程式，並開啟 `Tkinter` 儀表板，即時顯示從 Supabase 抓取的最新庫存數量。

---

## 🧠 Technical Details

### Action Recognition Model
核心視覺模型基於 `r3d_18` (3D ResNet-18)，採用 16 幀的滑動視窗 (Sliding Window) 進行推論。輸出的機率值會傳入基於規則的**狀態機 (State Machine)**，透過追蹤時序狀態並驗證 `Assemble` 步驟，將進度即時疊加顯示於攝影機畫面上。

### Inventory Dashboard
以 `Tkinter` 結合 `SupabaseREST` 開發，系統會自動輪詢 Supabase 資料庫。當特定組件庫存歸零時，UI 會動態切換色彩並呈現警示動畫。

---
<div align="center">
  <i>Developed at National Taiwan University of Science and Technology (NTUST)</i>
</div>
