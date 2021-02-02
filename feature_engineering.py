import emoji
import re
import string
from nltk.corpus import wordnet as wn


ascii_emoticons = [":‑)", ":)", ":-]", ":]", ":-3", ":3", ":->", ":> ", "8-)", "8)", ":-}", ":}", ":o)", ":c)", ":^)", "=]", "=)", ":‑D", ":D", "8‑D", "8D", "x‑D", "xD", "X‑D", "XD", "=D", "=3", "B^D", "c:", "C:", ":-))",
                   ":‑(", ":(", ":‑c", ":c", ":‑<", ":<", ":‑[", ":[", ":-||", ">:[", ":{", ":@", ":(", ";(", ":'‑(", ":'‑)", ":')", "D‑':", "D:<", "D:", "D8", "D;", "D=", "DX", ":‑O", ":O", ":‑o", ":o", ":-0", "8‑0", ">:O", ":‑P", ":P", "X‑P", "XP", "x‑p", "xp", ":‑p", ":p", ":‑Þ", ":Þ", ":‑þ", ":þ", ":‑b", ":b", "d:", "=p", ">:P"]


def score_tweet_rarity(tweet: str, dict):
    tweet = re.sub(r':[\S]+:', ' ', tweet)
    table_punctuation = str.maketrans(dict.fromkeys(string.punctuation))
    tweet = tweet.translate(table_punctuation)
    tweet_tokens = tweet.split()

    return sum([dict[word] for word in tweet_tokens]) / len(tweet_tokens)


def score_tweet_meanings(tweet: str):
    tweet = re.sub(r':[\S]+:', ' ', tweet)
    table_punctuation = str.maketrans(dict.fromkeys(string.punctuation))
    tweet = tweet.translate(table_punctuation)
    tweet_tokens = tweet.split()

    return sum([len(wn.synsets(word)) for word in tweet_tokens]) / len(tweet_tokens)


def repeated_char(token: str):
    for i in range(len(token) - 1):
        if token[i] == token[i + 1]:
            return True


def punctuation_or_repeated_letters(token: str):
    if repeated_char(token) or ("!" or "?" or ".." in token):
        return 2
    else:
        return 1


def score_tweet_lexical(tweet: str):
    tweet_tokens = tweet.split()

    return sum([punctuation_or_repeated_letters(word) for word in tweet_tokens]) / len(tweet_tokens)


def score_tweet_emoticons_or_emojis(tweet: str):
    if emoji.emoji_count(tweet) > 0:
        return True
    else:
        return any(emoticon in tweet for emoticon in ascii_emoticons)


def feature_engineering(df, tweet_column: str, inverted_frequency_dict: dict):
    # Spoken
    df['spoken'] = df[tweet_column].apply(
        lambda text: "*" in text or "-" in text)

    # Rarity
    df['rarity'] = df[tweet_column].apply(
        lambda text: score_tweet_rarity(text, inverted_frequency_dict))

    # Meanings
    df['meanings'] = df[tweet_column].apply(
        lambda text: score_tweet_meanings(text))

    # Lexical
    df['lexical'] = df[tweet_column].apply(
        lambda text: score_tweet_lexical(text))

    # Emoticons
    df['emoticon'] = df[tweet_column].apply(
        lambda text: score_tweet_emoticons_or_emojis(text))

    return df
