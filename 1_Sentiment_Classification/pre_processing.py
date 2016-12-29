# encoding: utf-8

import sys
import re
import pickle

'''
pre-process tweets, and build vocabulary

Run:

$ python pre_processing.py training.1600000.processed.noemoticon.csv testdata.manual.2009.06.14.csv
'''


# 提取文件中有用的字段
def pre_process(org_file, output_file):
    emoticons_str = r"""
            (?:
                [:=;] # Eyes
                [oO\-]? # Nose (optional)
                [D\)\]\(\]/\\OpP] # Mouth
            )"""

    regex_str = [
        emoticons_str,
        r'<[^>]+>',  # HTML tags
        r'(?:@[\w_]+)',  # @-mentions
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

        r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
        r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
        r'(?:[\w_]+)',  # other words
        r'(?:\S)'  # anything else
    ]

    tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
    emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)

    def tokenize(s):
        return tokens_re.findall(s)

    def preprocess(s, lowercase=True):
        tokens = tokenize(s)
        if lowercase:
            tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
        return " ".join(tokens)

    output = open(output_file, 'w', encoding="utf-8")
    with open(org_file, buffering=10000, encoding='latin-1') as f:
        try:
            for line in f:  # "4","2193601966","Tue Jun 16 08:40:49 PDT 2009","NO_QUERY","AmandaMarie1028","Just woke up. Having no school is the best feeling ever "
                line = line.replace('"', '')
                clf = line.split(',')[0]  # 4
                if clf == '0':
                    clf = [0, 0, 1]  # 消极评论
                elif clf == '2':
                    clf = [0, 1, 0]  # 中性评论
                elif clf == '4':
                    clf = [1, 0, 0]  # 积极评论

                tweet = line.split(',')[-1]
                outputline = str(clf) + ':%:%:%:' + preprocess(tweet) + '\n'
                output.write(
                    outputline)  # [0, 0, 1]:%:%:%: that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D
        except Exception as e:
            print(e)
    output.close()  # 处理完成，处理后文件大小127.5M


# 创建词汇表
def create_lexicon(train_file):
    lex = []
    with open(train_file, buffering=10000, encoding='latin-1') as f:
        try:
            count_word = {}  # 统计单词出现次数
            for line in f:
                line = line[:-1]  # remove \n
                tweet = line.split(':%:%:%:')[1]
                words = tweet.split(" ")
                for word in words:
                    if word not in count_word:
                        count_word[word] = 1
                    else:
                        count_word[word] += 1

            count_word = sorted(count_word.items(), key=lambda t: t[1], reverse=True)
            for word, word_count in count_word:
                if word_count < 50000 and word_count > 10:  # 过滤掉一些词
                    lex.append(word)
        except Exception as e:
            print(e)
    return lex


if __name__ == '__main__':
    train = sys.argv[1]
    test = sys.argv[2]

    pre_process(train, 'training.csv')
    pre_process(test, 'tesing.csv')

    lex = create_lexicon('training.csv')

    with open('lexcion.p', 'wb') as f:
        pickle.dump(lex, f)

    print("Total words: %s" % len(lex))

    print("The first 100 words are:\n%s" % (", ".join(lex[:100])))
    print("The last 100 words are: \n%s" % (", ".join(lex[-100:])))
