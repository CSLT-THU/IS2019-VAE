# v-vector-tf

Tensorflow and kaldi implementation of our interspeech paper "VAE-based regularization for deep speaker embedding"

## Dependency

1. computer
2. Linux (centos 7)
3. conda (Python 3.6)
4. Tensorflow-gpu 1.8
5. kaldi-toolkit

## Setting up the environment

1. [install kaldi](https://github.com/kaldi-asr/kaldi)
2. git clone the code and modify the `path.sh`, make sure that `path.sh` contains your kaldi path
3. create a conda environment and install the necessary Python package

```bash
# for example
conda create -n tf python=3.6
conda activate tf
pip install -r requirements.txt
```

## Datasets and X-vector

1. VoxCeleb
2. SITW
3. CSLT_SITW

## Steps

1. use kaldi to extract x-vector from uttrance and get `xvector.ark` files
2. covert the kaldi `xvector.ark` files to numpy binary data format (`xvector.ark` -> `xvector.npz`)
3. use tensorflow to train a model
4. use kaldi recipes to calculate EER

## About
