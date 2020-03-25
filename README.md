# NLPzoo
A collection of the most popular Natural Language Processing algorithms, frameworks and applications (inspired by tensorlayer/RLzoo).

## Folders

### Data (git ignored therefore not present here)
* `disaster-tweets.csv` contains tweets about real disasters and exaggerated 'fake' tweets which are not directly related to any disaster. This dataset is part of the "Real or Not? NLP with Disaster Tweets" Kaggle competition. [Download here](https://www.kaggle.com/c/nlp-getting-started/data)

* `shakespeare.txt` contains the complete texts of William Shakespeare. [Download here](https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt)
 
* `airline-tweets.csv` has tweets from 2015 from travellers expressing their feelings on their flying experience. [Download here] (https://www.kaggle.com/crowdflower/twitter-airline-sentiment)

* `cornell-movie-dialogs-corpus` is the classic Natural Language Processing training dataset. It contains 220,579 conversation exchanges. [Download here] (http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

* `ubuntu-dialogs-corpus` dialogs taken from online chat forums on the topic of Ubuntu. [Download here](http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/)

### ML (Machine Learning)
* This folder contains vanilla machine learning models classifiying sample text data. `ml_models.py` contains the main script used to compare the accuracy of the different models. Models include Naive Bayes and Support Vector Machines. Different vectorizers are also used with each type of model. So far this has not made a significant difference to the accuracy scores of a specific model. 

## TODO

The plan for the coming weeks is as follows. 
* Add more models to ML folder.
* Include Deep Learning models for different varieties of neural networks. 
* Demonstration and implementation of baseline LSTM model to compare with BERT
* Demonstration and implementation of the BERT language model
* Other models (order to be decided later): UNILM, MASS, BART

## Sources

* https://github.com/umerhasan17/mlds
* https://medium.com/@bedigunjit
