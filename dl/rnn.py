import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from ml.ml_helper_functions import sentence_preprocessing
from collections import defaultdict

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
    print(dataset[data_column][:5], dataset[processed_data_column][:5])


main()