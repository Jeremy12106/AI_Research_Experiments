
import numpy as np
import matplotlib.pyplot as plt

fc1=np.load('assets/ANN0.npy')
fc2=np.random.randn(128*10).reshape([128,10])

label=np.load('assets/mnistLabel.npy')
input=np.load('assets/mnist.npy')/255.0
input=np.reshape(input,[len(input),-1])

####################    Onehot Encode   #######################################

D=np.identity(10)[label]

#-------------------    End code    -------------------------------------------

acc = []

for ep in range(100):
    

####################    Forward Pass X1   #####################################
    
    X1 = np.dot(fc1.T,input.T).T
    # X1 = np.dot(input, fc1)
    
#-------------------    End code    -------------------------------------------
    
    
####################    Forward Pass A1   #####################################

    A1 = X1.copy()
    A1[A1<0]=0
    
#-------------------    End code    -------------------------------------------
    
    
####################    Forward Pass X2   #####################################
    
    X2 = np.dot(A1, fc2)
    
#---------------------------------------------End code
    
    
####################    Forward Pass A2   #####################################
    
    A2 = 1/(1+np.exp(-X2))
    
#-------------------    End code    -------------------------------------------
    
    
####################    Backward delta  #######################################
    
    delta = (D-A2)*(A2*(1-A2))
    
#-------------------    End code    -------------------------------------------
    
    
####################    Backward grad  #######################################
    
    grad = (-2)*(np.dot(A1.T,delta))
    
#-------------------    End code    -------------------------------------------
    

    fc2=fc2-0.05*grad
    
    choose=np.argmax(A2,1)
    
    a = np.sum(choose==label)/len(label)
    acc.append(a)
    print(str(ep)+": "+ str(a) )
 
plt.plot([i for i in range(100)], acc)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.savefig("results/Backpropagation.png", format='png')
plt.show()

