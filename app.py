import sys
from database.dbProcess import ForumDatabase
from strProcess import message_process
from dataProcess import data_process
from dataProcess import build_word_cloud
from dataProcess import get_text_from_clasters


def main(argv):
    database = ForumDatabase()
    messages = database.get_messages()
    messages = message_process(messages)
    y = data_process(messages)
    clusters_text = get_text_from_clasters(messages, y)
    build_word_cloud(clusters_text[0], "img1")
    build_word_cloud(clusters_text[1], "img2")


if __name__ == "__main__":
    main(sys.argv)
