# Chinese Speech Recognition

Use TensorFlow to train a Chinese Speech Recognition (ASR).

## Dataset

We use the open dataset from Dong Wang, Xuewei Zhang, Zhiyong Zhang called [THCHS30](http://data.cslt.org/thchs30/standalone.html), which could be used to a simple Automatic Speech Recognition (ASR) system for Chinese.

We can use the following method to download the full required dataset to the `data` folder:

```bash
$ sh download_data.sh
```

## Train

```
python train.py
```