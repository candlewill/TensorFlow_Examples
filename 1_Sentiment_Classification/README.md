# Sentiment Classification

This example use Neural Network (NN) and Convolutional NN to classify tweets into 3 classes, negative (0), neutral (2), positive (4).

## Data

1. Download from [Sentiment140](http://help.sentiment140.com/for-students/)

Data format:

    * 0 – the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
    * 1 – the id of the tweet (2087)
    * 2 – the date of the tweet (Sat May 16 23:58:44 UTC 2009)
    * 3 – the query (lyx). If there is no query, then this value is NO_QUERY.
    * 4 – the user that tweeted (robotickilldozr)
    * 5 – the text of the tweet (Lyx is cool)

In China, the file can be downloaded using Baidu Yun more fast from [here](http://pan.baidu.com/s/1jHCiTb4), psw: 4wub

2. Unzip the zip file, we can get two files:

    * training.1600000.processed.noemoticon.csv（238M）
    * testdata.manual.2009.06.14.csv（74K）

3.