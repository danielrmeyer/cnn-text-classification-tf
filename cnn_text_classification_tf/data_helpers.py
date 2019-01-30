import numpy as np
import re


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(data_files=[],
                         labels=[]):
    """
    This loads the data from files and cleans the text. Since the data files
    are stored by class, the labels can be added after reading in the 
    file.
    
    So the output might look something like this:
    ['some text', [0,0,1]]
    
    where [0,0,1] indicates class/label as a one-hot vector.
    """
    data_lengths = []
    x_text = []
    for file in data_files:
        x_temp = list(open(file, "r", encoding='utf-8').readlines())
        x_temp = [s.strip() for s in x_temp]
        data_lengths = data_lengths + [len(x_temp)]
        x_text = x_text + x_temp
    
    x_text = [clean_str(sent) for sent in x_text]
    
    # Generate labels
    labels_temp = []
    for i in range(0, len(labels)):
        labels_temp = labels_temp + [[labels[i] for _ in range(0, data_lengths[i])]]
    y = np.concatenate(labels_temp, 0)
    return [x_text, y]


def batch_iter(x, y, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(x)
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_x = x[shuffle_indices]
            shuffled_y = y[shuffle_indices]
        else:
            shuffled_x = x
            shuffled_y = y
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            batch_x = shuffled_x[start_index:end_index]
            batch_y = shuffled_y[start_index:end_index]
            # Reshape to match tensorflows expected input 
            # (batch, num_channels, sequencelength)
            batch_x = batch_x.reshape((batch_x.shape[0], batch_x.shape[2], batch_x.shape[1]))
            yield zip(batch_x, batch_y)
