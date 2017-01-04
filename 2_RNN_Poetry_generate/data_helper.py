from collections import Counter, defaultdict
import numpy as np

'''
Data load and pre-process
'''


def load_data(filename):
    contents = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            title, content = line.strip().split(":")
            contents.append(content)
    return contents


def build_vocab(texts, vocab_size):
    # chars count
    all_chars = []
    for text in texts:
        all_chars += list(text)
    counter = Counter(all_chars)
    word_tf_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*word_tf_pairs)

    # just use the top-n words according to word frequent
    words = words[:vocab_size]
    # word to ID
    word_idx = dict(zip(words, range(len(words))))
    # the index of OOV is 0
    word_idx = defaultdict(lambda: 0, word_idx)
    return word_idx


def vectorize_text(texts, vocab):
    to_num = lambda word: vocab[word]
    texts_vector = [list(map(to_num, list(text))) for text in texts]
    return texts_vector


def batch_iter(filename, batch_size, num_epochs, maxlen, step=1, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    texts = load_data(filename)
    vocab = build_vocab(texts, 10000)
    data = vectorize_text(texts, vocab)

    sentences = []
    next_sentences = []
    for sent_vec in data:
        sent_len = len(sent_vec)
        for i in range(0, sent_len-maxlen-1, step):
            start_index=i
            end_index=i+maxlen
            sentences.append(sent_vec[start_index: end_index])
            next_sentences.append(sent_vec[start_index+1: end_index+1])

    """
    sentences         next_sentences
    [6,2,4,6,9]       [2,4,6,9,9]
    [1,4,2,8,5]       [4,2,8,5,5]
    """

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    data = batch_iter("poetry.txt", 128, 16, 10, 1)
    for i in data:
        exit()
        print(i)
