
# 手刻卷積神經網路 (CNN)
此儲存庫包含一個手刻的卷積神經網路 (CNN) 實作，使用基本的程式庫如 NumPy 和 SciPy。模型包含卷積、激活以及全連接層，模仿簡單的 CNN 架構。

### 功能特色
- 卷積層：使用 scipy.signal.convolve2d 實現 2D 卷積。
- ReLU 激活函數：引入非線性，實作 ReLU 激活函數。
- 全連接層：兩個全連接層處理並分類從卷積層提取的特徵。
- Softmax 層：使用 Softmax 激活函數進行最終分類。
- 分層誤差計算：在每層輸出與預存的答案進行比較，確保每層的運作正確。

### 檔案說明
- assets/: 包含模型權重 (.npy 文件) 和執行 CNN - Homemade steelmaking 所需的輸入數據。
- CNN - Homemade steelmaking.py: 執行 CNN 架構的主要腳本，包含卷積層、全連接層以及誤差計算。

### 執行步驟
1. 克隆此儲存庫。
2. 確保 assets/ 資料夾中有必要的 .npy 權重和輸入數據文件。
3. 執行 CNN - Homemade steelmaking.py 腳本：  
    `python CNN.py`
4. 腳本將顯示每層的誤差計算結果與總運行時間。