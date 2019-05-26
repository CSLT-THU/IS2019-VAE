# Author: Yang Zhang
# Author: Xueyi Wang
# Apache 2.0.
# 2019, CSLT

import argparse
import numpy as np


def prepare_label_data(source_path, dest_path):
    print("source_path: ", source_path)
    print("dest_path: ", dest_path)
    print("start zip...")
    print("waiting...")

    utt2spk = np.loadtxt(source_path, dtype=bytes).astype(str)
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
                # print(j)
    spk = np.array(temp)
    spk = spk.reshape(-1, 1)

    spker = []
    for i in temp:
        if i not in spker:
            spker.append(i)

    spker = np.array(spker)

    print('spk', spk.shape)
    print('spker', spker.shape)

    np.savez(dest_path, spk_list=spk, spker=spker)

    print("prepare_label_data {} is done".format(dest_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", help="source_path of utt2spk")
    parser.add_argument("--dest_path", help="destination of spk.npz")
    args = parser.parse_args()

    source_path = args.source_path
    dest_path = args.dest_path

    prepare_label_data(source_path, dest_path)
