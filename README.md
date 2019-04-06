# v-vector-tf

Tensorflow implementation of our interspeech paper "VAE-based regularization for deep speaker embedding"

## Dependency

0. computer
1. Linux (centos)
2. conda (Python 3.6)
3. Tensorflow-gpu 1.8
4. kaldi-tookit

## Setting up the environment

1. install kaldi

2. create a conda environment and install the necessary Python package.

## Datasets and X-vector

1. VoxCeleb
2. SITW
3. CSLT_SITW

## Useage

1. Covert the kaldi vector files to numpy binary data format (ark->npz)

2. use tensorflow to train a model 

3. use kaldi recipes to calculate EER

## About
