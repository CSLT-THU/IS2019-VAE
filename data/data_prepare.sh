#!/bin/bash

# Author: Yang Zhang
# Author: Xueyi Wang
# Apache 2.0.
# 2019, CSLT

# xvector
for ark in `find -name "xvector.ark"`
do
	npz=`dirname $ark`"/xvector.npz"
	python -u zip.py \
		--source_path  $ark \
		--dest_path $npz 
	echo
done
echo

# utt2spk
for utt2spk in `find -name "utt2spk"`
do
	spknpz=`dirname $utt2spk`"/spk.npz"
	python -u label.py \
		--source_path  $utt2spk \
		--dest_path $spknpz
	echo
done
echo

echo data_prepare all DONE!
