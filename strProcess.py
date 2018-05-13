import string

import nltk as nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


stemmer = PorterStemmer()


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def string_prepare(str):
    str = str.lower()
    str = "".join([ch for ch in str if ch not in string.punctuation])
    tokens = nltk.word_tokenize(str)
    stems = stem_tokens(tokens, stemmer)
    return stems


def del_stop():
    stop = set(stopwords.words('english'))
    sentence = "this is a foo bar sentence"
    print([i for i in sentence.lower().split() if i not in stop])


# print(string_prepare("DD,!? sd, s ,E"))



corpus = ["This is very strange",
          "This is very nice",
          "My name is nice",
          "My name is Vadim and I`d like to play cs go",
          "This is very prety girl"]
corpus1 = ["My name is nice"]
vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus)
test = vectorizer.transform(corpus1)
k = 2
model = KMeans(n_clusters=k, random_state=0, n_jobs = -2)
model.fit(X)
y = model.predict(X)
text = ""
text1 = ""
for idx,claster in enumerate(y):
    print(claster, corpus[idx])
    if claster == 0:
        text += corpus[idx] + " "
    if claster == 1:
        text1 += corpus[idx] + " "


wordcloud = WordCloud(background_color='white',
                          width=1200,
                          height=1000
                      ).generate(text)
                         # ).generate(" ".join(corpus))


plt.imshow(wordcloud)
plt.axis('off')
plt.show()
wordcloud = WordCloud(background_color='white',
                          width=1200,
                          height=1000
                      ).generate(text1)
                         # ).generate(" ".join(corpus))


plt.imshow(wordcloud)
plt.axis('off')
plt.show()
