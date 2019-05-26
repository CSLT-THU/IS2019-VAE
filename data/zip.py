# Author: Yang Zhang
# Author: Xueyi Wang
# Apache 2.0.
# 2019, CSLT

import argparse
import os
import numpy as np


def ark2npz(source_path, dest_path):
    print("source_path: ", source_path)
    print("dest_path: ", dest_path)
    print("start zip...")
    print("waiting...")

    count = 0
    labels = []
    vector = []
    with open(source_path) as f:
        lines = f.readlines()
        for line in lines:
            count += 1
            # print("load {} success!".format(count))
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

            vector.append(num_list)
    labels = np.array(labels, dtype="<U64")

    vector = np.array(vector, dtype="float64")

    print("vector shape:")
    print(labels.shape)
    print(vector.shape)

    np.savez(dest_path, vector=vector, utt=labels)

    print("sucessfully convert {} to {} ".format(source_path, dest_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", help="source_path of xvector(ark)")
    parser.add_argument("--dest_path", help="destination of xvector(npz)")
    args = parser.parse_args()

    source_path = args.source_path
    dest_path = args.dest_path

    ark2npz(source_path, dest_path)
