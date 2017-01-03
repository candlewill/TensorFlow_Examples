# Sentiment Classification

This example use Neural Network (NN) and Convolutional NN (CNN) to classify tweets into 3 classes, negative (0), neutral (2), positive (4).

## Data

1. Download from [Sentiment140](http://help.sentiment140.com/for-students/)

 Data format:

 * 0 – the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
 * 1 – the id of the tweet (2087)
 * 2 – the date of the tweet (Sat May 16 23:58:44 UTC 2009)
 * 3 – the query (lyx). If there is no query, then this value is NO_QUERY.
 * 4 – the user that tweeted (robotickilldozr)
 * 5 – the text of the tweet (Lyx is cool)

 In China, the file can be downloaded faster using Baidu Yun storage from [here](http://pan.baidu.com/s/1jHCiTb4) with psw *4wub*

2. Unzip the zip file, we can get two csv files:

 * training.1600000.processed.noemoticon.csv (238M)
 * testdata.manual.2009.06.14.csv (74K)

## Data pre-processing

Pre-processing each tweet and convert the two raw data into two files: training.csv, testing.csv, which only contains label and tweet. The file *lexicon.p* is the vocabularies.

```python
python pre-processing.py training.1600000.processed.noemoticon.csv testdata.manual.2009.06.14.csv
```

After that, we can get **34,963** words, and the first and last 100 words are:

The first 100 words are:

```text
time, u, back, am, from, will, :, 2, what, one, we, im, about, know, really, can, amp, don't, can't, see, had, its, some, still, how, ', want, night, new, think, (, *, home, ), there, thanks, when, miss, more, if, as, 3, they, much, need, here, off, well, hope, then, an, last, tomorrow, /, has, been, or,
 morning, great, twitter, her, again, feel, sad, oh, haha, he, wish, fun, why, only, sleep, bad, right, happy, would, very, i'll, tonight, come, did, make, them, by, sorry, getting, gonna, though, way, better, over, she, nice, wait, watching, 4, should, could, that's, bed
```

The last 100 words are:

```text
pues, colombian, @pippi43, genitals, ock, dunny, tthe, woodpecker, @destinyhope92, jiggle, @katelyntarver, @thebrandbuilder, idt, sufjan, @flyaarmy, @edwintcg, @pixie_maw, kickstart, johnsons, torrenting, @sanya29, http://thrdl.es/, pigtails, stalls, holey, tui, @mrsfudgecrumpet, bram, @thanr, @meekakitt
y, irma, retailer, @saurik, haylie, scampi, lastest, @eliiiiza, janitor, ahhhhhhhhhhhhhhhhhh, @lidles, com's, bucky, fuccin, registers, @little_lin, herald, loos, @davidrules04, lolcat, noi, everythinggg, mallu, tole, swiped, interupted, @kevinhart4real, loaned, @charlii1, paragon, jenkins, @springwesten
d, guerrilla, @_scene_queen_, maybes, mercurial, nags, #bfd, tins, @megdia, popo, minh, @_lauren_mallory, @gcrush, barbershop, inflict, newscast, monstrous, dodged, unfortunatelly, @kimbalicious, embarking, dsm, soree, dawsons, excelente, locator, picasso, #the, @jksgirlx2, progression, powershot, rowe,
@nelsonmaud, simulation, sensual, hhhh, youtuber, beacause, morninq, underbelly
```

## Neural Networks

Train a neural network to classify the tweets:

```python
python nn_classify.py
```

## CNN
Train a CNN to classify the tweets:

```python
python cnn_classify.py
```