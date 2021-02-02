import emoji

url_token = "__URL__"
user_token = "__USER__"


def replace_urls(df, column: str):
    df[column] = df[column].str.replace(
        r'http\S+|www.\S+', url_token, case=False)


def replace_user(df, column: str):
    df[column] = df[column].str.replace(
        r'@[^\s]+', user_token, case=False)


def to_lowercase(df, column: str):
    df[column] = df[column].str.lower()


def demojize(df, column: str):
    df[column] = df[column].apply(emoji.demojize)


def replace_xml_escaped_characters(df, column: str):
    df[column] = df[column].replace(
        ['&apos', '&quot', "&lt", "&amp", "&gt"], '')


def clean(df, tweet_column: str):
    df = df.dropna()
    df = df.drop_duplicates()
    replace_urls(df, tweet_column)
    replace_user(df, tweet_column)
    to_lowercase(df, tweet_column)
    demojize(df, tweet_column)
    replace_xml_escaped_characters(df, tweet_column)

    return df
