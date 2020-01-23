import numpy as np
import pandas as pd
import sys, os
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


def main():
    eda()
    sentence_preprocessing(dataset, data_column, processed_data_column)
    train_x, test_x, train_y, test_y = model_selection.train_test_split(dataset[processed_data_column], dataset[target], test_size=0.3)
    (train_x_vectorized, test_x_vectorized) = vectorize(TfidfVectorizer(max_features=5000), dataset, processed_data_column, train_x, test_x)
    # todo pad sequences


main()