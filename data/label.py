# Author: Yang Zhang
# Mail: zyziszy@foxmail.com
# Apache 2.0.
# 2019, CSLT

import numpy as np

# label
utt2spk = np.loadtxt('./utt2spk', dtype=bytes).astype(str)
all_labels = []
for i in utt2spk:
    all_labels.append(i[1].strip('id'))

spker = []
for i in all_labels:
    if i not in spker:
        spker.append(i)
spk = []

temp = []
for i in all_labels:
    for j in range(len(spker)):
        if spker[j] == i:
            temp.append(j)
            print(j)
spk = np.array(temp)
spk = spk.reshape(-1, 1)

spker = []
for i in temp:
    if i not in spker:
        spker.append(i)

spker=np.array(spker)

print('spk', spk.shape)
print('spker',spker.shape)
print("start zip...")
np.savez('./spk.npz', spk_list=spk,spker=spker)
