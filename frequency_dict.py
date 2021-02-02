import re
import collections
import string


def invert_value(a: int):
    if a == 1:
        return 10
    elif a == 2:
        return 5
    elif a == 3:
        return 2
    else:
        return 1


def create_frequency_dict(df, text_column: str):
    text = ' '.join(df[text_column])
    text = re.sub(r':[\S]+:', ' ', text)

    table_punctuation = str.maketrans(dict.fromkeys(string.punctuation))
    text = text.translate(table_punctuation)
    text = " ".join(text.split())

    wordlist = text.split()
    frequency_dict = collections.Counter(wordlist)

    return frequency_dict


def inverted_frequency_dict(frequency_dict: dict):
    freq_inverted = dict((k, invert_value(v))
                         for k, v in frequency_dict.items())
    return freq_inverted


def create_inverted_frequency_dict(df, text_column: str):
    frequency_dict = create_frequency_dict(df, text_column)
    return inverted_frequency_dict(frequency_dict)
