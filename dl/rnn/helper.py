import unidecode
import string
import random
import re
import time 
import math
import torch
from torch.autograd import Variable

all_characters = string.printable


def import_and_sanitize(filepath):
    n_characters = len(all_characters)
    current_file = unidecode.unidecode(open(filepath).read())
    file_len = len(current_file)
    print("successfully imported file {} with length = {}", filepath, file_len)
    return current_file, n_characters

# Turn string into list of longs
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)

def random_chunk(current_file, chunk_len):
    file_len = len(current_file)
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return current_file[start_index:end_index]

def random_training_set(current_file, chunk_len):    
    chunk = random_chunk(current_file, chunk_len)
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)