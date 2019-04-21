# Author: Yang Zhang
# Mail: zyziszy@foxmail.com
# Apache 2.0.
# 2019, CSLT


import os
import json
import numpy as np
path = './xvector.ark'

def ark2npz(path):
    count = 0
    labels = []
    train = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            count += 1

            print("load {} success!".format(count))
            line.strip('\n')
            vector_string = ""
            id = ""
            is_vector = False
            for c in line:
                if c == '[':
                    is_vector = True
                if is_vector:
                    if c != '[' and c != ']':
                        vector_string += c
                if (not is_vector) and c != " ":
                    id += c
            labels.append(id)
            num_list = vector_string.split(' ')
            num_list.pop()
            del(num_list[0])
            num_list = list(map(eval, num_list))

            train.append(num_list)
    labels = np.array(labels, dtype="<U64")

    train = np.array(train, dtype="float64")

    print("start zip...")

    print(labels.shape)
    print(train.shape)

    np.savez('./a', vector=train, label=labels)

    print("zip is done")

if __name__=="__main__":
    ark2npz(path)
