import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


def data_process(messages, k=2):
    vectorizer = TfidfVectorizer(min_df=1)
    x = vectorizer.fit_transform(messages)
    model = KMeans(n_clusters=k, random_state=0, n_jobs=-2)
    model.fit(x)
    y = model.predict(x)
    return y


def get_text_from_clasters(corpus, y):
    text = []
    for idx, claster in enumerate(y):
        try:
            text[claster] += corpus[idx] + " "
            print()
        except IndexError:
            text.insert(claster, corpus[idx] + " ")
    return text


def build_word_cloud(text_arr, img_name):
    fig = plt.figure()
    for i in range(0, len(text_arr)):
        wordcloud = WordCloud(background_color='white',
                              width=1200,
                              height=1000
                              ).generate(text_arr[i])
        fig.add_subplot(1, 2, i + 1)
        plt.imshow(wordcloud)
        plt.axis('off')
    plt.savefig(img_name)
    plt.show(block=True)
