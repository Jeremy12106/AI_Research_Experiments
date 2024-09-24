
import time
import numpy as np
from scipy import signal

mask=np.load('assets/SCMK_W0.npy')
mask=np.transpose(mask,[2,3,0,1])
mask=np.save('assets/masksSample.npy',mask)
mask=np.load('assets/masksSample.npy')
mask=mask[0,:,::-1,::-1]

b1=np.load('assets/SCMK_W1.npy')
FC1=np.load('assets/SCMK_W2.npy')
bFC1=np.load('assets/SCMK_W3.npy')
FC2=np.load('assets/SCMK_W4.npy')
bFC2=np.load('assets/SCMK_W5.npy')

ans0=np.load('assets/SCMK_F0.npy')
ans1=np.load('assets/SCMK_F1.npy')
ans2=np.load('assets/SCMK_F2.npy')
ans3=np.load('assets/SCMK_F3.npy')
ans4=np.load('assets/SCMK_F4.npy')

ans0=np.transpose(ans0,[0,3,1,2])
ans0=ans0[0]
ans2=ans2[0]

input=np.load('assets/input.npy')
# cv2.imshow('test',(input*255).astype(np.uint8) )
# cv2.waitKey(0)

L1out=[]

start = time.time()

#########################  Convolution   Code  ################################

for i in range(len(mask)):
    z = signal.convolve2d(input, mask[i], mode = 'same')              
    z = z+b1[i]
    z[z<0]=0
    L1out.append(z)

L1out=np.array(L1out)

#------------------------  End Code   -----------------------------------------


L1out=np.array(L1out)
print('\nError in the First Layer:')
print(np.sum(np.abs(ans0-L1out)))

L2out=np.transpose(L1out,[1,2,0]).reshape([-1])
print('\nError in the Second Layer:')
print(np.sum(np.abs(ans2-L2out)))

#########################  pooling   Code  ########################################

# 假設每個 feature map 是 2D，進行 2x2 最大池化
# pooled_L1out = []
# for feature_map in L1out:
#     pooled_map = []
#     h, w = feature_map.shape
#     for i in range(0, h, 2):
#         row = []
#         for j in range(0, w, 2):
#             row.append(np.max(feature_map[i:i+2, j:j+2]))
#         pooled_map.append(row)
#     pooled_L1out.append(pooled_map)

# pooled_L1out = np.array(pooled_L1out)

# 更新 L2out 的輸出來自於池化後的 L1out
# L1out = pooled_L1out

#------------------------  End Code   -----------------------------------------

#########################  FC1   Code  ########################################

L3out = np.dot(L2out, FC1) + bFC1
L3out[L3out<0]=0

#------------------------  End Code   -----------------------------------------


print('\nError in the FC1 Layer:')
print(np.sum(np.abs(ans3[0]-L3out)))


#########################  FC2   Code  ########################################

L4out = np.dot(L3out, FC2) + bFC2
L4out = np.exp(L4out)/sum(np.exp(L4out))

#------------------------  End Code   -----------------------------------------


print('\nError in the FC2 Layer:')
print(np.sum(np.abs(ans4[0]-L4out)))

end = time.time()
print('\ntime: ', end-start)

