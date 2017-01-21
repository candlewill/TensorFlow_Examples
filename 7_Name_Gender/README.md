# Gender classification

Gender classification according to Chinese Name.

## Dataset

The [data](https://pan.baidu.com/s/1hsHTEU4) is crawled from internet. There are 351,791 names in the dataset labeled as male and female. Note that a name could be labeled as male and female at the same time, as some Chinese names is neutral and suit for both.

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

## Decoding

```
python decode.py
```

## Test Result

```
白富美 男
高帅富 男
王婷婷 男
田野 男
何云超 男
张天 男
何云雷 男
王苗苗 女
王津 男
王培英 男
李开复 男
彭丽媛 女
习近平 女
周小川 男
范冰冰 男
赵丽颖 女
周杰伦 男
杨丽 男
刘诗诗 女
刘德华 男
宋小宝 男
郭德纲 女
王欧 男
杨颖 男
```