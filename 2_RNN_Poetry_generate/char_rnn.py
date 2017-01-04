from .data_helper import batch_iter
# import tensorflow as tf

'''
Use Char_RNN to generate Chinese Tang Poetry.
'''


def run():
    # training parameters
    batch_size = 128
    num_epochs = 100
    texts = batch_iter("poetry.txt", batch_size, num_epochs)

    # Data



if __name__ == '__main__':
    run()