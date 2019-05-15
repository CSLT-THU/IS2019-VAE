#!/bin/bash
# Copyright 2019    Yang Zhang
# Apache 2.0.


python -u main.py \
	--epoch 200 \
	--batch_size 200 \
	--n_hidden 1800 \
	--learn_rate 0.00001 \
	--beta1 0.5 \
	--dataset_path ./data/voxceleb_combined_200000/xvector.npz \
	--z_dim 200 \
	--KL_weigth 0.03 \
	--cohesive_weight 0 \
	--is_training 1 

