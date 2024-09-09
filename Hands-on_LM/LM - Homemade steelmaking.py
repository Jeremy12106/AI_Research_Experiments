
import numpy as np
import jieba
import time
import numpy as np
import os


########################   Read text ##################################

path = 'assets/data.txt'
f = open(path, 'r',encoding="utf-8")
input=f.read()
a=input.split()
f.close()
words_list=[]

wordsSp = jieba.cut(input)

for word in wordsSp:
    if (word !='\n')&(word!='')&(word!=' ')&(word!='※'):
        words_list.append(word)
        
#-----------------------------------------------------------------------   


book={}
book2={}
ibook={}
ibook2={}
for i in range(len(words_list)-2):
    word=words_list[i]
    nextword=words_list[i+1]
    nextnextword=words_list[i+2]
    
    ########################   Fill in books ##################################
        
    if word not in book:
        book[word]=dict()
        book[word][nextword]=1
    else:
        if nextword not in book[word]:
            book[word][nextword]=1
        else:
            book[word][nextword]+=1
    
    if word not in book2:
        book2[word]=dict()
        book2[word][nextnextword]=1
    else:
        if nextnextword not in book2[word]:
            book2[word][nextnextword]=1
        else:
            book2[word][nextnextword]+=1
    
    #-----------------------------------------------------------------------  
    

## two words
      
start='我'
ans=start
ptr=start

stop='。'
stopCount=3

for i in range(100000):#   Generate 100000 !
    
    
    if i <1: 
        val=list(book[ptr].values())
        X=np.cumsum(val)
        pick=np.random.randint(X[0],X[-1]+1, size=10)[0]
        choose=np.where(pick<=X)[0][0]
        ans+=list(book[ptr].keys())[choose]
        ptr=list(book[ptr].keys())[choose]
        
    else:
        a=list(book[ptr].keys())
        pa=[]
        pda1=[]
        pda2=[]
        for j in range(len(a)):
                
            ########################   Calcuate P(A)  ##############################
            
            pa = [book[ptr][nextword] / sum(book[ptr].values()) for nextword in list(book[ptr].keys())]
            
            #---------------------------------------------------------------------
            
            
            ########################   Calcuate P(d1|A)  ###########################
            
            pda1 = [book2[ptr].get(nextnextword, 0) / sum(book2[ptr].values()) for nextnextword in list(book[ptr].keys())]
            
            #-------------------------------------------------------------------------
            
            
            ########################   Calcuate Calcuate P(d2|A)   ##################
            
            pda2 = [book2[nextword].get(nextnextword, 0) / sum(book2[nextword].values()) for nextword in list(book[ptr].keys())]
            
            #----------------------------------------------------------------------
        

        pa=np.array(pa)
        pda1=np.array(pda1)
        pda2=np.array(pda2)
        pad=pa*pda1*pda2
        
           
    
        ################  choose the action from p(a|d) distribution ################

        next_index = np.random.choice(len(pad))
        ptr = a[next_index]
        ans += ptr

        #-------------------------------------------------------------------------
        

    if ptr == '。': 
        stopCount -= 1  

    if stopCount == 0:  
        break


# 指定要儲存的檔案名稱
file_name = "results/" + time.strftime("%Y%m%d_%H%M%S") + ".txt"
os.makedirs(os.path.dirname(file_name), exist_ok=True)
# 開啟檔案並寫入字串
with open(file_name, "w", encoding="utf-8") as file:
    file.write(ans)

print(f"檔案已經成功儲存到 {file_name}。")
print(ans)
