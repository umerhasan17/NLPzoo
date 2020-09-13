A selection of very brief snippets of knowledge to help with understanding research papers. 

## Embeddings

"An embedding is a relatively low-dimensional space into which you can translate high-dimensional vectors." 
A good embedding "captures some of the semantics of the input by placing semantically similar inputs close together in the embedding space".
For example, embeddings for the words toothbrush and toothpaste should be close together.

Usually in NLP, word embeddings are used however there can be emoji embeddings, sentence embeddings (also called encodings) etc. 
Word embeddings encode words into a vector representation.

* **One-hot vector** embeddings simply represent n words with an n dimensional vector with all 0s and one 1 at the index of the word.
* **SVD based embeddings** rely on iterating over a dataset and store data in a matrix X. 
Then the matrix is decomposed into the USV<sup>T</sup> format, with the rows of U as the word embeddings.
* **Contextualized word embeddings** generate vectors for words depending on the context. 
This helps to differentiate vectors for a word that can have multiple meanings, for example light, bark, left etc.
* **Learned embeddings** are generated using a model (e.g. a neural network). 
The embeddings are the weights of the network and are adjusted to minimize loss on the task. Example [here](https://colab.research.google.com/notebooks/mlcc/intro_to_sparse_data_and_embeddings.ipynb).
* **Neural embeddings** are simply embeddings that are the weights of a neural network. One example is the continuous bag of words model.
* **Positional embeddings** were introduced alongside Transformers. 
They are designed to recover position information as this is lost because of the way the Transformers are designed.
Therefore they store both position information and embedding information.
  * **Learned positional embeddings** are learned vectors containing same information as positional embeddings.
  * **Sinusoidal positional embeddings** generate embeddings using the sin/cosine functions. 
  
**TODO** add further information on embeddings
* Word2Vec & GloVe generate 1 vector for 1 word.
* Graph embeddings
* Skip gram
* Pre-trained libraries (spaCy, FastText)
* Embedding out of vocab words
* Universal encoder
* Huggingface embedding functions


## Language Models

* **Probabilistic Language Modeling** 

https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf

# Sources

https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture
http://web.stanford.edu/class/cs224n/
