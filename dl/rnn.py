import numpy as np
import pandas as pd
import sys, os
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from ml.ml_helper_functions import sentence_preprocessing, vectorize
from collections import defaultdict
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

dataset = pd.read_csv('data/airline-tweets.csv')
data_column = 'text'
processed_data_column = 'processed_text'
target = 'airline_sentiment'

def eda():
    print(dataset.head())
    # seeing how many values of each category we want to predict are there in the data
    print(dataset[target].value_counts())


def data_prep():
    eda()
    sentence_preprocessing(dataset, data_column, processed_data_column)
    train_x, test_x, train_y, test_y = model_selection.train_test_split(dataset[processed_data_column], dataset[target], test_size=0.3)
    (train_x_vectorized, test_x_vectorized) = vectorize(TfidfVectorizer(max_features=5000), dataset, processed_data_column, train_x, test_x)
    # todo pad sequences


class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # why is there a super call here?
        super(VanillaRNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i20 = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 128
rnn = VanillaRNN(26, n_hidden, 2)