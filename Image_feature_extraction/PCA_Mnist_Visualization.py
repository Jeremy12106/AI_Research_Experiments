
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

input2=input[-10000:]
D2=D[-10000:]
input=input[0:-10000]
D=D[0:-10000]
label2=label[-10000:]
label=label[0:-10000]

plt.figure(figsize=(15,10))
plt.subplot(231)

X=input.copy().astype(float)
cov=np.cov(X.T)
val,vec=np.linalg.eig(cov)
principle=np.argsort(-val)[0:2]
dim2=np.dot(X,vec[:,principle])

Alpha=1
size=1
for cl in range(10):
    chos=dim2[np.argwhere(label==cl).reshape([-1])]
    plt.scatter(chos[:,0],chos[:,1],label=str(cl),alpha=Alpha,marker='x',s=size)
plt.legend()
plt.title('Input Component')

input=255*(input-np.min(input,1).reshape([-1,1]))/(np.max(input,1).reshape([-1,1])-np.min(input,1).reshape([-1,1]))

plt.imshow(input[0].reshape([28,28]).astype(np.uint8))

##################################

grab=input[0:50].copy()
grab=grab.reshape([5,10,28,28]).transpose([0,2,1,3])
grab=np.reshape(grab,[5*28,10*28])

plt.imshow(grab.astype(np.uint8))

###################################

step=100

image=input.reshape([-1,28,28])
image=(image-np.min(image))/(np.max(image)-np.min(image))*255
A=image[0]
B=image[1]

delta=(B-A)/step

steps=np.arange(step)
steps=steps*delta.reshape([28,28,1])

steps=steps.reshape([28,28,10,10])

vis01=steps.reshape([28,28,10,10]).transpose([2,0,3,1]).reshape([280,280]).astype(np.uint8)

plt.imshow(vis01)

###################################

dim=20
X=input.copy().astype(float)
cov=np.cov(X.T)
val,vec=np.linalg.eig(cov)

b=vec[:,0:dim]
c=np.dot(X,b)
bInv=np.dot(np.linalg.inv(np.dot(b.T,b)),b.T)

IA=c[41]
IB=c[42]

delta=(IB-IA)/step

steps=np.arange(step)
steps=steps*delta.reshape([dim,1])
steps=steps+IA.reshape([dim,1])

d=np.dot(steps.T,bInv)

print(np.sum(val[0:dim])/np.sum(val))

d=(d-np.min(d))/(np.max(d)-np.min(d))*255
d=d.real

d=d.reshape([10,10,28,28]).transpose([0,2,1,3]).reshape([280,280]).astype(np.uint8)
plt.imshow(d)


rec=[]
rec2=[]
EP=100

for ep in range(EP):
    
    X1=np.dot(fc1.T,input2.T).T

    A1=X1.copy()
    A1[A1<0]=0

    X2=np.dot(fc2.T,A1.T).T 
    A2=1/(1+np.exp(-X2))    
    choose2=np.argmax(A2,1)
    
########################################Forward Pass X1=???
    X1=np.dot(fc1.T,input.T).T
########################################End code
    

########################################Forward Pass A1=???
    A1=X1.copy()
    A1[A1<0]=0
########################################End code
    
    
########################################Forward Pass X2=???
    X2=np.dot(fc2.T,A1.T).T
########################################End code
    
    
########################################Forward Pass A2=???
    A2=1/(1+np.exp(-X2))
########################################End code
    
    
########################################Backward delta=???
    delta=(D-A2)*(A2)*(1-A2)
########################################End code
    
    
########################################Backward grad=???
    grad=-2*np.dot(A1.T,delta)+0.001*fc2
########################################End code
    

    fc2=fc2-0.05*grad
    
    
    choose=np.argmax(A2,1)
    print(str(ep)+" TraningACC: "+str(np.sum(choose==label)/len(label))+' TestingACC:'+str(np.sum(choose2==label2)/len(label2)))
    rec.append(np.sum(choose==label)/len(label))

    # print(str(ep)+": "+str(np.sum(choose2==label2)/len(label2)))
    rec2.append(np.sum(choose2==label2)/len(label2))
     
plt.plot(np.arange(EP),rec,label='TrainingACC')
plt.plot(np.arange(EP),rec2,label='TestingACC')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('ACC')

plt.figure(figsize=(15,10))
plt.subplot(231)




