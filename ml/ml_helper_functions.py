from collections import defaultdict

import numpy as np
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def sentence_preprocessing(Corpus, column_name, output_column_name):
    # Step - a : Remove blank rows if any.
    Corpus[column_name].dropna(inplace=True)
    # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    Corpus[column_name] = [entry.lower() for entry in Corpus[column_name]]
    # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
    Corpus[column_name]= [word_tokenize(entry) for entry in Corpus[column_name]]
    # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(Corpus[column_name]):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        Corpus.loc[index, output_column_name] = str(Final_words)
    return Corpus

def vectorize(Vectorizer, Corpus, column, Train_X, Test_X):
    Vectorizer.fit(Corpus[column])
    Train_X_Vectorized = Vectorizer.transform(Train_X)
    Test_X_Vectorized = Vectorizer.transform(Test_X)
    return (Train_X_Vectorized, Test_X_Vectorized)