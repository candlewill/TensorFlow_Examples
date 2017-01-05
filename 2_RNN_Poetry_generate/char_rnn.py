from data_helper import batch_iter
# import tensorflow as tf

'''
Use Char_RNN to generate Chinese Tang Poetry.
'''


def run():
    # training parameters
    batch_size = 128
    num_epochs = 100
    texts = batch_iter("poetry.txt", 128, 16, 10, 1, 10)
    for text in texts:
        x_batch, y_batch = zip(*text)
        print(x_batch)
        print(y_batch)

    # Data



if __name__ == '__main__':
    run()