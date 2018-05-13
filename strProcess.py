import string
import nltk as nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def string_prepare(sentence: str):
    stemmer = PorterStemmer()
    sentence = sentence.lower()
    sentence = "".join([ch for ch in sentence if ch not in string.punctuation])
    tokens = nltk.word_tokenize(sentence)
    stems = stem_tokens(tokens, stemmer)
    str_to_return = ""
    for stem in stems:
        str_to_return += stem + " "
    return str_to_return


def del_stop_words(sentence):
    stop = set(stopwords.words('english'))
    sentence_to_return = ""
    for i in sentence.lower().split():
        if i not in stop:
            sentence_to_return += i + " "
    return sentence_to_return


def message_process(messages):
    correct_messages = []
    for message in messages:
        message = del_stop_words(message)
        message = string_prepare(message)
        correct_messages.insert(len(correct_messages), message)
    return correct_messages

#
#
#
# corpus = ["This is very strange",
#           "This is very nice",
#           "My name is nice",
#           "My name is Vadim and I`d like to play cs go",
#           "This is very prety girl"]
# corpus1 = ["My name is nice"]
# vectorizer = TfidfVectorizer(min_df=1)
# X = vectorizer.fit_transform(corpus)
# k = 2
# model = KMeans(n_clusters=k, random_state=0, n_jobs = -2)
# model.fit(X)
# y = model.predict(X)
# text = ""
# text1 = ""
# for idx,claster in enumerate(y):
#     print(claster, corpus[idx])
#     if claster == 0:
#         text += corpus[idx] + " "
#     if claster == 1:
#         text1 += corpus[idx] + " "


# wordcloud = WordCloud(background_color='white',
#                           width=1200,
#                           height=1000
#                       ).generate(text)
#                          # ).generate(" ".join(corpus))
#
# plt.imshow(wordcloud)
# plt.axis('off')
# plt.show()

# wordcloud = WordCloud(background_color='white',
#                           width=1200,
#                           height=1000
#                       ).generate(text1)
#                          # ).generate(" ".join(corpus))
#
#
# plt.imshow(wordcloud)
# plt.axis('off')
# plt.show()
