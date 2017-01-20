# Gender classification

Gender classification according to Chinese Name.

## Dataset

The data is crawled from internet. There are 351,791 names in the dataset labeled as male and female. Note that a name could be labeled as male and female at the same time, as some Chinese names is neutral and suit for both.

The format of dataset:

```
姓名,性别
熊猫哥,男
周笑冉,女
毛丹璎,女
郭展成,男
...
```

## Train

```
python train.py
```