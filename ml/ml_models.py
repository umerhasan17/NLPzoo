import pandas as pd
from sklearn import model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from ml_helper_functions import sentence_preprocessing, vectorize


def main():
    data_column = 'text'
    processed_column = 'text_final'
    target = 'target'

    print("Preprocessing...")
    Corpus = sentence_preprocessing(pd.read_csv('../data/disaster-tweets.csv'), data_column, processed_column)
    Vectorizers = [TfidfVectorizer(max_features=5000), CountVectorizer()]
    Vectorizer_Columns = ["tfidf", "count"]
    Models = [naive_bayes.MultinomialNB(), svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'), LogisticRegression()]

    print("Splitting data...")
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus[processed_column],Corpus[target],test_size=0.3)
    for Model in Models:
        for index, Vectorizer in enumerate(Vectorizers):
            Corpus[Vectorizer_Columns[index]] = Corpus[processed_column]
            print("Vectorizing", "...")
            (Train_X_Vectorized, Test_X_Vectorized) = vectorize(Vectorizer, Corpus, Vectorizer_Columns[index], Train_X, Test_X)
            print("Generating predictions", "...")
            score = generate_predictions(Model, Train_X_Vectorized, Test_X_Vectorized, Train_Y, Test_Y)
            print(Model, " with vectorizer ", Vectorizer_Columns[index] , " Accuracy Score -> ", score)

def generate_predictions(Model, Train_X_Vectorized, Test_X_Vectorized, Train_Y, Test_Y):
    Model.fit(Train_X_Vectorized,Train_Y)
    predictions = Model.predict(Test_X_Vectorized)
    score = accuracy_score(predictions, Test_Y)*100
    return score  


main()
