import numpy as np
from scipy import signal
import time
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image


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

# 位元的範圍
bit_range = range(4, 13)

# 儲存量化誤差的列表
quantization_errors = []

for bit in bit_range:
    
    # 計算權重的最大值和最小值
    top=np.max(mask)
    bot=np.min(mask)
    
    #############    Calculate   weight   Quantization   ######################

    #############    Calculate   A   ######################
    
    # 計算量化因子 A
    A = (2**bit-1)/(top-bot)
    
    #------------    End   A    ---------------------------

    #############    Calculate   qw   #####################
    
    # 將權重歸一化至指定位元數
    wnorm=A*mask-A*bot
    # 轉換為16位元整數
    qw=wnorm.astype(np.uint32)
    
    #----------------    End   Calculate   qw   -----------

    #----------------    End   Quantize weight    -----------------------------
    

    # 計算輸入的最大值和最小值
    top2=np.max(input)
    bot2=np.min(input)
    

    #############    Calculate   input Quantization   #########################

    #############    Calculate   B   ######################
    
    # 計算量化因子 B
    B = (2**bit-1)/(top2-bot2)
    
    #------------    End   B    ---------------------------

    #############    Calculate   qx   #####################
    
    # 將輸入歸一化至指定位元數
    xnorm=B*input-B*bot2
    # 轉換為16位元整數
    qx=xnorm.astype(np.uint32)
    
    #----------------    End   Calculate   qx   -----------


    # 將 bot 重複擴充為 3x3 的矩陣
    bot=np.repeat(bot,3*3).reshape([3,3])
    # 將 bot2 重複擴充為 28x28 的矩陣
    bot2=np.repeat(bot2,28*28).reshape([28,28])
    

    #############    Calculate   Term 1 int Convolution   #####################
    
    # 初始化 Term 1
    term1=np.zeros([len(mask),28,28])
    
    # 計算 Term 1 的卷積
    for i in range(128):
        term1[i] = signal.convolve2d(qx,qw[i],mode='same')
        
    # 將 Term 1 歸一化
    term1=(1/(A*B))*term1
    
    #----------------    End   Term 1    --------------------------------------
    

    print('term1 Error :')
    print(np.mean(np.abs(term1Ans-term1)))
    

    #############    Calculate   Term 2 qBiasD   ##############################
    
    # 初始化 Term 2
    term2=np.zeros([len(mask),28,28])

    # 計算 Term 2
    term2=np.tile(signal.convolve2d(qx,bot/B.reshape([1,1]),mode='same').reshape([-1]),128).reshape([128,28,28])

    #----------------    End   Term 2    --------------------------------------
    

    print('term2 Error :')
    print(np.mean(np.abs(term2Ans-term2)))
    

    #############    Calculate   Term 3 qBias   ###############################
    
    # 初始化 Term 3
    term3=np.zeros([len(mask),28,28])

    # 計算 Term 3
    term3 = np.tile(signal.convolve2d(bot2,bot,mode='same').reshape([-1]),128).reshape([128,28,28])
    
    #----------------    End   Term 3    --------------------------------------
    

    print('term3 Error :')
    print(np.mean(np.abs(term3Ans-term3)))
    

    #############    Calculate   Term 4    ####################################
    
    # 初始化 Term 4
    term4=np.zeros([len(mask),28,28])
    
    # 計算 Term 4 的卷積
    for i in range(128):
        term4[i] = signal.convolve2d(bot2/A.reshape([1,1]),qw[i],mode='same')
    
    #----------------    End   Term 4    -----------------------
    

    print('term4 Error :')
    print(np.mean(np.abs(term4Ans-term4)))
    
    L1out=[]
    
    
    # 計算L1out
    L1out=term1+term2+term3+term4+np.repeat(b1,28*28).reshape([-1,28,28])
    L1out[L1out<0]=0

    min=np.min([L1out,ans0])
    max=np.max([L1out,ans0])
    
    # 歸一化
    L1outNorm=((L1out-min)/(max-min)*255)
    ansNorm=((ans0-min)/(max-min)*255)
    print(str(bit)+'bit - quantize Error:')
    
    # 計算量化誤差
    Loss=np.mean(np.abs(L1outNorm-ansNorm))
    print(np.mean(np.abs(L1out-ans0)))
    
    print("--------------------")
    
    # 將Loss加入量化誤差的列表
    quantization_errors.append(Loss)


# 繪製最終量化誤差之XY圖，
# X軸為量化之bit取樣數(4位元至12位元)，其Y軸為量化誤差
plt.plot(bit_range, quantization_errors, marker='o')
plt.xlabel('Quantization Bit')
plt.ylabel('Quantization Error')
plt.title('Quantization Error vs. Quantization Bit')
plt.savefig('results/Quantization_Analysis.png', format='png')
plt.show()