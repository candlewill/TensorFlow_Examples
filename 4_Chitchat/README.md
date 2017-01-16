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

1. Remove / , and group into question and answer

 python pre_process.py

2. Build vocabulary, and convert texts into vecotrs according to their index

 python vectorizer.py

The generated files: train_encode.vec and train_decode.vec are used for training, whose vocabulary is train_encode_vocabulary and train_decode_vocabulary, respectively.

## Train

 We use [seq2seq model](https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/seq2seq_model.py) on the training data. It will take a few hours.

 python train.py

 After about 589,500 steps, we get the following result:

```
587000 0.000305144 2.56671503615
0 12.543833172375583
1 23.283587395237618
2 20.160063524352733
3 14.362551288543015
587500
587500 0.000305144 2.59014926028
0 8.500202516716408
1 16.222189910741516
2 19.48035661490604
3 16.742266002404577
588000
588000 0.000295989 2.58287223005
0 10.108112100344743
1 15.532271967787727
2 21.85774124267814
3 16.732469326616826
588500
588500 0.000295989 2.60331172109
0 10.217336668387876
1 15.90760132489462
2 20.418236969921555
3 17.01274752262256
589000
589000 0.00028711 2.57389337921
0 8.111649969765645
1 17.354193061727415
2 18.568189964633955
3 15.888793235232294
589500
589500 0.00028711 2.57721918631
```

The training loss is 2.57, and the PPL is about 15.

## Decoding

 Use the trained model to similate chat bot:

 python decoding.py


## Testing result
