from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

phrases = [ 'aluminium is a light material', 
            'the sun emits light']

one_hot_vectorizer = CountVectorizer(binary=True)
one_hot = one_hot_vectorizer.fit_transform(phrases).toarray()
sns.heatmap(one_hot, annot=True, cbar=False)

plt.show()