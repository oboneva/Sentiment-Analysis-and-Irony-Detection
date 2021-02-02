import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json

from tweet_preprocess import clean
from feature_engineering import feature_engineering

sns.set()

classifiers = [MultinomialNB(), SVC(), RandomForestClassifier(),
               KNeighborsClassifier(n_neighbors=3), DecisionTreeClassifier()]


def parse_file(filepath: str):
    bike_sharing = pd.read_csv(filepath)
    return bike_sharing


def modify_classes(df):
    df["class"] = df["class"].replace(['figurative', 'irony', 'sarcasm'], True)
    df["class"] = df["class"].replace(['regular'], False)


def preprocess_file(filepath):
    df = parse_file(filepath)
    df = df.rename(columns={"tweets": "tweet"})
    df = clean(df, "tweet")
    modify_classes(df)

    return df


def merge_data():
    test_dataframe = parse_file("./Datasets/test.csv")
    test_dataframe = test_dataframe.rename(columns={"tweets": "tweet"})
    test_dataframe = clean(test_dataframe, "tweet")
    test_dataframe = test_dataframe.dropna()
    modify_classes(test_dataframe)
    test_dataframe.to_csv("./Datasets/test_cleaned.csv")

    train_dataframe = parse_file("./Datasets/train.csv")
    train_dataframe = train_dataframe.rename(columns={"tweets": "tweet"})
    train_dataframe = clean(train_dataframe, "tweet")
    train_dataframe = train_dataframe.dropna()
    modify_classes(train_dataframe)
    train_dataframe.to_csv("./Datasets/train_cleaned.csv")

    merged_dataframe = pd.concat([train_dataframe, test_dataframe])
    merged_dataframe.to_csv("./Datasets/merged.csv")


def plot(correct, predicted):
    mat = confusion_matrix(correct, predicted)

    sns.heatmap(mat.T, square=True, annot=True, fmt="d",
                xticklabels=["True", "False"], yticklabels=["True", "False"])

    plt.xlabel("true labels")
    plt.ylabel("predicted label")

    plt.show()


def classify(x_train, x_test, y_train, y_test, model):
    model.fit(x_train.to_numpy(), y_train.to_numpy())
    predicted_categories = model.predict(x_test.to_numpy())

    # plot(y_test.to_numpy(), predicted_categories)

    print(classification_report(y_test.to_numpy(), predicted_categories))


def balanced_classes(df, class_column):
    df_positives = df[df[class_column] == True]
    df_negatives = df[df[class_column] == False]

    df_positives = df_positives[:len(df_negatives.index)]
    df_balanced = pd.concat([df_positives, df_negatives])

    return df_balanced


def classify_with_tfidf(df):
    test_dataframe = preprocess_file("./Datasets/test.csv")
    train_dataframe = preprocess_file("./Datasets/train.csv")
    df = pd.concat([train_dataframe, test_dataframe])

    df_balanced = balanced_classes(df, "class")
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    kf = KFold(n_splits=10, shuffle=True)

    for train_indicies, test_indicies in kf.split(df_balanced):
        x_train = df_balanced.iloc[train_indicies]["tweet"]
        x_test = df_balanced.iloc[test_indicies]["tweet"]
        y_train = df_balanced.iloc[train_indicies]["class"]
        y_test = df_balanced.iloc[test_indicies]["class"]

        classify(x_train=x_train, x_test=x_test,
                 y_train=y_train, y_test=y_test, model=model)


def classify_with_custom_features():
    test_dataframe = preprocess_file("./Datasets/test.csv")
    train_dataframe = preprocess_file("./Datasets/train.csv")
    df = pd.concat([train_dataframe, test_dataframe])

    with open('./dict_inverted.json') as f:
        freq_inverted = json.load(f)
        f.close()

    df = feature_engineering(df, "tweet", freq_inverted)

    df_balanced = balanced_classes(df, "class")
    model = make_pipeline(MultinomialNB())
    kf = KFold(n_splits=10, shuffle=True)

    for train_indicies, test_indicies in kf.split(df_balanced):
        x_train = df_balanced.iloc[train_indicies][[
            'spoken', "rarity", "meanings", "lexical", "emoticon"]]
        x_test = df_balanced.iloc[test_indicies][[
            'spoken', "rarity", "meanings", "lexical", "emoticon"]]
        y_train = df_balanced.iloc[train_indicies]["class"]
        y_test = df_balanced.iloc[test_indicies]["class"]

        classify(x_train=x_train, x_test=x_test,
                 y_train=y_train, y_test=y_test, model=model)


def main():
    classify_with_custom_features()


if __name__ == "__main__":
    main()
