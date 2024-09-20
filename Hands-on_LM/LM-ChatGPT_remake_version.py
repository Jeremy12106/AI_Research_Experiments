import numpy as np
import jieba
import time
import os
from collections import defaultdict

#-----------------LM - ChatGPT_remake_version-------------------------#

########################   Read text ##################################

path = 'assets/data.txt'
with open(path, 'r', encoding="utf-8") as f:
    input_txt = f.read()

# 過濾符號
words_list = list(filter(lambda x: x not in ['\n', '', ' ', '※', '…', '」', '「', '—'], jieba.cut(input_txt)))

#-----------------------------------------------------------------------   

book = defaultdict(lambda: defaultdict(int))
book2 = defaultdict(lambda: defaultdict(int))

for i in range(len(words_list) - 2):
    word = words_list[i]
    nextword = words_list[i + 1]
    nextnextword = words_list[i + 2]

    book[word][nextword] += 1
    book2[word][nextnextword] += 1

#---------------- Generate text ----------------#

start = input("輸入一個開頭字元: ")
ans = start
ptr = start

stop = '。'
stopCount = int(input("輸入文章段落數(句號): "))
print("生成文章中")

max_repeats = 10  # 其他詞的最大出現次數
blacklist_max_repeats = 1  # 黑名單詞的最大出現次數
word_counts = defaultdict(int)

# 設定黑名單
blacklist = {'啊', '啊啊', '啊啊啊', '喂', '喂喂', '住手', '我的', '我的，', '你的'}  # 在這裡填入你想限制的詞

for i in range(100000):
    if i < 1:
        if ptr in book and book[ptr]:  # 確保 ptr 有對應的條目
            val = np.array(list(book[ptr].values()))
            X = np.cumsum(val)
            pick = np.random.randint(X[0], X[-1] + 1)
            choose = np.searchsorted(X, pick)
            ptr = list(book[ptr].keys())[choose]
            ans += ptr
            word_counts[ptr] += 1  # 更新計數
        else:
            print("沒有可用的下個字元，生成結束。")
            break
    else:
        if ptr in book and book[ptr]:  # 確保 ptr 有對應的條目
            nextwords = list(book[ptr].keys())
            valid_nextwords = [
                word for word in nextwords 
                if (word not in blacklist and word_counts[word] < max_repeats) or 
                   (word in blacklist and word_counts[word] < blacklist_max_repeats)
            ]
            if not valid_nextwords:
                print("所有候選字元都已達到最大出現次數，生成結束。")
                break

            pa = np.array([book[ptr][nextword] / sum(book[ptr].values()) for nextword in valid_nextwords])
            pda1 = np.array([book2[ptr].get(nextword, 0) / sum(book2[ptr].values()) for nextword in valid_nextwords])
            pda2 = np.array([book2[nextword].get(nextword, 0) / sum(book[nextword].values()) for nextword in valid_nextwords])

            pad = pa * pda1 * pda2
            
            if pad.sum() == 0:
                next_index = np.random.choice(len(valid_nextwords))  # 隨機選擇一個字元
            else:
                pad /= pad.sum()
                next_index = np.random.choice(len(pad), p=pad)

            ptr = valid_nextwords[next_index]
            ans += ptr
            word_counts[ptr] += 1  # 更新計數

            # 檢查是否遇到 stop 字元
            if ptr == stop:  
                stopCount -= 1

            if stopCount == 0:
                break
        else:
            print("沒有可用的下個字元，生成結束。")
            break


# 儲存生成文字
file_name = "results/" + time.strftime("%Y%m%d_%H%M%S") + ".txt"
os.makedirs(os.path.dirname(file_name), exist_ok=True)
with open(file_name, "w", encoding="utf-8") as file:
    file.write(ans)

print(f"檔案已經成功儲存到 {file_name}。")
print(ans)
