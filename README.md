# v-vector-tf

Tensorflow and kaldi implementation of our interspeech paper "VAE-based regularization for deep speaker embedding"

## Dependency

1. computer
2. Linux (centos)
3. conda (Python 3.6)
4. Tensorflow-gpu 1.8
5. kaldi-tookit

## Setting up the environment

1. install kaldi

2. create a conda environment and install the necessary Python package

## Datasets and X-vector

1. VoxCeleb
2. SITW
3. CSLT_SITW

## Useage

1. use kaldi to extract x-vector from uttrance (xvector.ark)

2. Covert the kaldi vector files to numpy binary data format (ark->npz)

3. use tensorflow to train a model

4. use kaldi recipes to calculate EER

## About
