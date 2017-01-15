# Chitchat Bot

This project use Tensorflow to make a simple chitchat bot in Chinese. The chatbot use RNN model, and is trained on Chinese corpus.

## Dataset

Chinese conversation corpus from this [repo](https://github.com/rustch3n/dgk_lost_conv).

1. Download dataset

`wget https://raw.githubusercontent.com/rustch3n/dgk_lost_conv/master/dgk_shooter_min.conv.zip`

2. Unzip the zip file, we get a `dgk_shooter_min.conv` file, move it to `data` folder. The format of `conv` file is:

```
.conv 格式:
//M 表示话语，E 表示分割。
E
M 话语 a
M 话语 b
M 话语 c
M 话语 d
E
M 话语 a
M 话语 b
M 话语 c
M 话语 d
```

For example

```
E
M 畹/华/吾/侄/
M 你/接/到/这/封/信/的/时/候/
M 不/知/道/大/伯/还/在/不/在/人/世/了/
E
M 咱/们/梅/家/从/你/爷/爷/起/
M 就/一/直/小/心/翼/翼/地/唱/戏/
M 侍/奉/宫/廷/侍/奉/百/姓/
M 从/来/不/曾/遭/此/大/祸/
M 太/后/的/万/寿/节/谁/敢/不/穿/红/
M 就/你/胆/儿/大/
M 唉/这/我/舅/母/出/殡/
```

## Pre-process

