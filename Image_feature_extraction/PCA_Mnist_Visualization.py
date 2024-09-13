import numpy as np
import matplotlib.pyplot as plt

# 載入模型參數和數據
fc1 = np.load('assets/ANN0.npy')
fc2 = np.random.randn(128 * 10).reshape([128, 10])

label = np.load('assets/mnistLabel.npy')
input = np.load('assets/mnist.npy') / 255.0
input = np.reshape(input, [len(input), -1])

# Onehot編碼
D = np.identity(10)[label]

# 分割數據集
input2 = input[-10000:]
D2 = D[-10000:]
input = input[:-10000]
D = D[:-10000]
label2 = label[-10000:]
label = label[:-10000]

# 創建圖形窗口
plt.figure(figsize=(15, 10))

# 子圖1: 顯示降維後的數據分佈
plt.subplot(231)
X = input.copy().astype(float)
cov = np.cov(X.T)
val, vec = np.linalg.eig(cov)
principle = np.argsort(-val)[:2]
dim2 = np.dot(X, vec[:, principle])

Alpha = 1
size = 1
for cl in range(10):
    chos = dim2[np.argwhere(label == cl).reshape([-1])]
    plt.scatter(chos[:, 0], chos[:, 1], label=str(cl), alpha=Alpha, marker='x', s=size)
plt.legend()
plt.title('Input Component')
plt.savefig('results/pca_analyze.png', bbox_inches='tight')

# 顯示第一張圖像
plt.figure(figsize=(15, 10))
plt.subplot(232)
input_image = 255 * (input - np.min(input, 1).reshape([-1, 1])) / (np.max(input, 1).reshape([-1, 1]) - np.min(input, 1).reshape([-1, 1]))
plt.imshow(input_image[0].reshape([28, 28]).astype(np.uint8))
plt.title('First Image')
plt.savefig('results/first_img.png', bbox_inches='tight')

# 顯示50張圖像的網格
plt.figure(figsize=(15, 10))
plt.subplot(233)
grab = input[:50].copy()
grab = grab.reshape([5, 10, 28, 28]).transpose([0, 2, 1, 3])
grab = np.reshape(grab, [5 * 28, 10 * 28])
grab = (grab - np.min(grab)) / (np.max(grab) - np.min(grab)) * 255
plt.imshow(grab.astype(np.uint8), cmap='gray')
plt.title('Image Grid')
plt.savefig('results/50Grids.png', bbox_inches='tight')

# 顯示圖像間的過渡
plt.figure(figsize=(15, 10))
plt.subplot(234)
step = 100
image = input.reshape([-1, 28, 28])
image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
A = image[41]
B = image[42]
delta = (B - A) / step
steps = np.arange(step)
steps = steps * delta.reshape([28, 28, 1])
steps = steps.reshape([28, 28, 10, 10])
vis01 = steps.reshape([28, 28, 10, 10]).transpose([2, 0, 3, 1]).reshape([280, 280]).astype(np.uint8)
plt.imshow(vis01)
plt.title('Transition Between Images')
plt.savefig('results/transition.png', bbox_inches='tight')

# 顯示主成分分析 (PCA) 後的圖像
plt.figure(figsize=(15, 10))
plt.subplot(235)
dim = 20
X = input.copy().astype(float)
cov = np.cov(X.T)
val, vec = np.linalg.eig(cov)
b = vec[:, :dim]
c = np.dot(X, b)
bInv = np.dot(np.linalg.inv(np.dot(b.T, b)), b.T)
IA = c[41]
IB = c[42]
delta = (IB - IA) / step
steps = np.arange(step)
steps = steps * delta.reshape([dim, 1])
steps = steps + IA.reshape([dim, 1])
d = np.dot(steps.T, bInv)
d = (d - np.min(d)) / (np.max(d) - np.min(d)) * 255
d = d.real
d = d.reshape([10, 10, 28, 28]).transpose([0, 2, 1, 3]).reshape([280, 280]).astype(np.uint8)
plt.imshow(d)
plt.title('PCA Components')
plt.savefig('results/pca_transition.png', bbox_inches='tight')

# 訓練神經網路
rec = []
rec2 = []
EP = 100

for ep in range(EP):
    X1 = np.dot(fc1.T, input2.T).T
    A1 = X1.copy()
    A1[A1 < 0] = 0
    X2 = np.dot(fc2.T, A1.T).T
    A2 = 1 / (1 + np.exp(-X2))
    choose2 = np.argmax(A2, 1)
    
    # Forward Pass
    X1 = np.dot(fc1.T, input.T).T
    A1 = X1.copy()
    A1[A1 < 0] = 0
    X2 = np.dot(fc2.T, A1.T).T
    A2 = 1 / (1 + np.exp(-X2))
    
    # Backward Pass
    delta = (D - A2) * (A2) * (1 - A2)
    grad = -2 * np.dot(A1.T, delta) + 0.001 * fc2
    fc2 = fc2 - 0.05 * grad
    
    choose = np.argmax(A2, 1)
    print(f"{ep} TrainingACC: {np.sum(choose == label) / len(label)} TestingACC: {np.sum(choose2 == label2) / len(label2)}")
    rec.append(np.sum(choose == label) / len(label))
    rec2.append(np.sum(choose2 == label2) / len(label2))

# 繪製訓練和測試準確度曲線
plt.figure(figsize=(15, 10))
plt.subplot(236)
plt.plot(np.arange(EP), rec, label='TrainingACC')
plt.plot(np.arange(EP), rec2, label='TestingACC')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('ACC')
plt.title('Accuracy Over Epochs')
plt.savefig('results/accuracy.png', bbox_inches='tight')

# plt.show()
