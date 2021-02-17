import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import re
import collections
from heapq import nlargest
import random


# import nltk
# nltk.download('stopwords')


def is_valid(token: str):
    if len(token) < 4:
        return False
    elif "." in token:
        return False
    elif re.search(r'\d+', token):
        return False
    else:
        return True


def to_text(df, text_column_name: str):
    text = ". ".join(df[text_column_name])
    text = text.lower()
    tokens = word_tokenize(text)
    stop_tokens = set(stopwords.words('english') + list(string.punctuation))
    candidates = [
        token for token in tokens if token not in stop_tokens and is_valid(token)]

    # print(len(candidates)) -> 15277

    phrases = []
    phrase = []
    for token in tokens:
        if token in stop_tokens or not is_valid(token):
            # and len(phrase) < 4 not in the original algo
            if len(phrase) > 1 and len(phrase) < 4:
                phrases.append(phrase)
            phrase = []
        else:
            phrase.append(token)

    frequency_dict = collections.Counter(candidates)
    # print(frequency_dict)

    degree_dict = {}
    for token in list(dict.fromkeys(candidates)):
        for phrase in phrases:
            if token in phrase:
                if token in degree_dict:
                    degree_dict[token] += len(phrase)
                else:
                    degree_dict[token] = len(phrase)

    # print(degree_dict)

    word_score_dict = {}
    for token in list(dict.fromkeys(candidates)):
        if token not in frequency_dict or token not in degree_dict:
            word_score_dict[token] = 0
        else:
            word_score_dict[token] = degree_dict[token] / frequency_dict[token]

    score_dict = {}
    for phrase in phrases:
        score_dict[" ".join(phrase)] = sum(
            [word_score_dict[token] for token in phrase]) / len(phrase)  # / len(phrase) not in the original algo

    # print(score_dict)

    key_phrases = nlargest(
        len(score_dict) // 3, score_dict, key=score_dict.get)

    key_phrases = random.sample(key_phrases, 20)

    print(key_phrases)

    return key_phrases


def clean_reviews():
    reviews_df = pd.read_csv("reviews.csv")
    to_text(reviews_df, "comment")


def main():
    clean_reviews()


if __name__ == "__main__":
    main()
