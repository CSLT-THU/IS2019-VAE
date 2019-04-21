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

check = []
for i in all_labels:
    if i not in check:
        check.append(i)
spk = []

temp = []
for i in all_labels:
    for j in range(len(check)):
        if check[j] == i:
            temp.append(j)
            print(j)
spk = np.array(temp)
spk = spk.reshape(-1, 1)

check = []
for i in temp:
    if i not in check:
        check.append(i)

check=np.array(check)

print('spk', spk.shape)
print('check',check.shape)
print("start zip...")
np.savez('./spk.npz', spk=spk,check=check)
