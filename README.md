# v-vector-tf

Tensorflow and kaldi implementation of our interspeech paper [VAE-based regularization for deep speaker embedding](https://github.com/zyzisyz/v-vector-tf/paper.pdf)

## Dependency

1. computer
2. Linux (centos 7)
3. conda (Python 3.6)
4. Tensorflow-gpu 1.8
5. kaldi-toolkit

## Datasets and X-vector

1. VoxCeleb
2. SITW
3. CSLT_SITW

## Steps

1. use kaldi to extract x-vector from uttrance and get `xvector.ark` files
2. covert the kaldi `xvector.ark` files to numpy binary data format (`xvector.ark` -> `xvector.npz`)
3. use tensorflow to train a VAE model, and get the V-vectors
4. use kaldi recipes to calculate EER (equal error rate)

## Useage

1. [install kaldi](https://github.com/kaldi-asr/kaldi) (note: if you are one of cslt member, you can referance[Dr. tzy's Kaldi](https://github.com/tzyll/kaldi))

2. create a conda environment and install the necessary Python package

```bash
# for example
conda create -n tf python=3.6
conda activate tf
pip install -r requirements.txt
```

3. git clone the code and modify the `path.sh`, make sure that `path.sh` contains your kaldi path

4. Download x-vector data from CSLT server

```bash
cd data
sh download.sh
```

## About

### Repo Author

Yang Zhang (zyziszy@foxmail.com)

### Contributors

- [@Lilt](166.111.134.19:8081/lilt/)
- [@fatejessie](https://github.com/fatejessie)
- [@xDarkLemon](https://github.com/xDarkLemon)
- [@AlanXiuxiu](https://github.com/AlanXiuxiu)
