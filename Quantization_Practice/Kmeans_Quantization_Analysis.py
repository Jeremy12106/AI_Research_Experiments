
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
from scipy import signal
from sklearn.cluster import KMeans


mask=np.load('assets/SCMK_W0.npy') # 載入權重
mask=np.transpose(mask,[2,3,0,1])   # 將權重的維度重新排列
mask=np.save('assets/masksSample.npy',mask)    # 儲存重新排列後的權重
mask=np.load('assets/masksSample.npy') # 載入重新排列後的權重
mask=mask[0,:,::-1,::-1]    # 取出特定維度的部分

b1=np.load('assets/SCMK_W1.npy')
ans0=np.load('assets/SCMK_F0.npy')
ans0=np.transpose(ans0,[0,3,1,2])[0]
input=np.load('assets/input.npy')

term1Ans=np.load('assets/term1Ans.npy')
term2Ans=np.load('assets/term2Ans.npy')
term3Ans=np.load('assets/term3Ans.npy')
term4Ans=np.load('assets/term4Ans.npy')


# 定義KMeans的量化範數
cluster_range = range(10, 50, 5)


# 儲存量化誤差的列表
kmeans_quantization_errors = []

for n_clusters in cluster_range:
    
    # 對權重進行KMeans量化
    mask_flatten = mask.reshape(-1, 1)  # 將權重展平
    kmeans_w = KMeans(n_clusters=n_clusters).fit(mask_flatten)
    qw_kmeans = kmeans_w.cluster_centers_[kmeans_w.labels_]
    qw_kmeans = qw_kmeans.reshape(mask.shape)  # 重新變回原來的形狀
    
    # 重新計算 bot，取量化後權重的最小值
    bot = np.min(qw_kmeans, axis=(1, 2))  # 計算每一個通道的最小值
    bot = np.repeat(bot, 3*3).reshape([len(bot), 3, 3])  # 將其擴充為 3x3 的矩陣
    
    # 對輸入進行KMeans量化
    input_flatten = input.reshape(-1, 1)  # 將輸入展平
    kmeans_x = KMeans(n_clusters=n_clusters).fit(input_flatten)
    qx_kmeans = kmeans_x.cluster_centers_[kmeans_x.labels_]
    qx_kmeans = qx_kmeans.reshape(input.shape)  # 重新變回原來的形狀

    # 重新計算 bot2，取量化後輸入的最小值
    bot2 = np.min(qx_kmeans)
    bot2 = np.repeat(bot2, 28*28).reshape([28, 28])  # 擴充為 28x28 的矩陣

    # 計算卷積
    term1_kmeans = np.zeros([len(mask), 28, 28])
    for i in range(128):
        term1_kmeans[i] = signal.convolve2d(qx_kmeans, qw_kmeans[i], mode='same')

    # 計算Term1的誤差
    print(f'term1 KMeans Error for {n_clusters} clusters:')
    print(np.mean(np.abs(term1Ans - term1_kmeans)))

    # 計算Term 2的卷積，這裡的 bot 基於 K-means 量化的結果
    term2_kmeans = np.tile(signal.convolve2d(qx_kmeans, bot[i] / np.max(bot[i]).reshape([1, 1]), mode='same').reshape([-1]), 128).reshape([128, 28, 28])
    print(f'term2 KMeans Error for {n_clusters} clusters:')
    print(np.mean(np.abs(term2Ans - term2_kmeans)))

    # 計算Term 3和Term 4類似，根據KMeans結果進行卷積
    term3_kmeans = np.tile(signal.convolve2d(bot2, bot[i], mode='same').reshape([-1]), 128).reshape([128, 28, 28])
    print(f'term3 KMeans Error for {n_clusters} clusters:')
    print(np.mean(np.abs(term3Ans - term3_kmeans)))

    term4_kmeans = np.zeros([len(mask), 28, 28])
    for i in range(128):
        term4_kmeans[i] = signal.convolve2d(bot2 / np.max(bot2).reshape([1, 1]), qw_kmeans[i], mode='same')
    print(f'term4 KMeans Error for {n_clusters} clusters:')
    print(np.mean(np.abs(term4Ans - term4_kmeans)))

    # 計算L1out
    L1out_kmeans = term1_kmeans + term2_kmeans + term3_kmeans + term4_kmeans + np.repeat(b1, 28*28).reshape([-1, 28, 28])
    L1out_kmeans[L1out_kmeans < 0] = 0

    min_val = np.min([L1out_kmeans, ans0])
    max_val = np.max([L1out_kmeans, ans0])

    L1outNorm_kmeans = ((L1out_kmeans - min_val) / (max_val - min_val) * 255)
    ansNorm_kmeans = ((ans0 - min_val) / (max_val - min_val) * 255)

     # 顯示每次量化後的結果
    # plt.figure(figsize=(10, 5))

    # 顯示L1outNorm_kmeans
    # plt.subplot(1, 2, 1)
    # plt.imshow(L1outNorm_kmeans[0], cmap='gray')
    # plt.title(f'L1out - {n_clusters} clusters (KMeans)')
    # plt.colorbar()

    # 顯示ansNorm_kmeans
    # plt.subplot(1, 2, 2)
    # plt.imshow(ansNorm_kmeans[0], cmap='gray')
    # plt.title(f'ansNorm - {n_clusters} clusters (KMeans)')
    # plt.colorbar()

    # plt.show()

    # 計算KMeans的量化誤差
    kmeans_loss = np.mean(np.abs(L1outNorm_kmeans - ansNorm_kmeans))
    print(f'{n_clusters}-cluster KMeans quantization Error:')
    print(np.mean(np.abs(L1out_kmeans - ans0)))

    print("--------------------")
    
    # 將KMeans量化誤差加入列表
    kmeans_quantization_errors.append(kmeans_loss)

# 繪製KMeans量化誤差的XY圖
plt.plot(cluster_range, kmeans_quantization_errors, marker='o', color='red')
plt.xlabel('KMeans Clusters')
plt.ylabel('Quantization Error')
plt.title('KMeans Quantization Error vs. Number of Clusters')
plt.savefig('results/KMeans_Quantization_Analysis.png', format='png')
plt.show()
