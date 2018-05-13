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
    cluster1_text = ""
    cluster2_text = ""
    text = []
    for idx, claster in enumerate(y):
        if claster == 0:
            cluster1_text += corpus[idx] + " "
        if claster == 1:
            cluster2_text += corpus[idx] + " "
    text.insert(1, cluster1_text)
    text.insert(2, cluster2_text)
    return text


def build_word_cloud(text, img_name):
    wordcloud = WordCloud(background_color='white',
                          width=1200,
                          height=1000
                          ).generate(text)
    plt.imshow(wordcloud)
    plt.savefig(img_name)
    plt.axis('off')
    plt.show()
