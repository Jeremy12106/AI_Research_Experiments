
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

af=np.random.randn(10000)

bit=8

##################     Linear Quantization Code    ############################

## Find the max Value and the min Value
MAX = max(af)
MIN = min(af)

## Quantization Code->transfer the type to np.uint8  !!
qa = (2**bit)*(af-MIN)/(MAX-MIN)
qa=np.uint8(qa)

#-----------------     END  Linear Quantization   ------------------------------


deqa=0


##################     Linear DeQuantization Code    ##########################

deqa = qa*(MAX-MIN)/(2**bit)+MIN

#-----------------     END  Linear DeQuantization   ---------------------------


plt.figure()
plt.hist(af,500,label='float',density=True,color='r')
plt.legend()
plt.savefig('results/Before_Quantization.png', format='png')
plt.show()


plt.figure()
plt.hist(deqa,500,label='quantization result',density=True,color='b')
plt.legend()
plt.savefig('results/Linear_Quantization.png', format='png')
plt.show()


afShape=np.reshape(af,[-1,1])


##################     Kmeans Quantization Training Code Here    ##############

kmeans = KMeans(n_clusters=2**bit)
kmeans.fit(afShape)

#-----------------     END  Training    ---------------------------------------


##################     Kmeans Quantization Code    ############################

## Predict The Group
code = kmeans.predict(afShape)

## Load Code Book
codebook = kmeans.cluster_centers_

## Find the Value (Dequantization)
pre = codebook[code]

#-----------------     END  Kmeans Quantization    ----------------------------


plt.figure()
plt.hist(pre,500,label='kmeans quantization result',density=True,color='g')
plt.legend()
plt.savefig('results/Kmeans_Quantization.png', format='png')
plt.show()


print('\nLinear Qunatization Error:')
print(np.mean(np.abs(af-deqa)))
print('\nKmeans Qunatization Error:')
print(np.mean(np.abs(afShape-pre)))