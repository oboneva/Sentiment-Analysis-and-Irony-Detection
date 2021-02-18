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
import joblib

from tweet_preprocess import clean
from feature_engineering import feature_engineering

sns.set()

classifiers = [MultinomialNB(), SVC(), RandomForestClassifier(),
               KNeighborsClassifier(n_neighbors=3), DecisionTreeClassifier()]
classifiers_names = ["naive_bayes", "svm", "random_forest",
                     "knn3", "decision_tree"]


def parse_file(filepath: str):
    df = pd.read_csv(filepath)
    return df


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


def plot(correct, predicted, filename):
    mat = confusion_matrix(correct, predicted)

    sns.heatmap(mat.T, square=True, annot=True, fmt="d",
                xticklabels=["True", "False"], yticklabels=["True", "False"])

    plt.xlabel("True labels")
    plt.ylabel("Predicted labels")

    plt.savefig(filename)
    plt.clf()


def classify(x_train, x_test, y_train, y_test, model, filename):
    model.fit(x_train.to_numpy(), y_train.to_numpy())
    predicted_categories = model.predict(x_test.to_numpy())

    plot(y_test.to_numpy(), predicted_categories, f"{filename}.png")

    report = classification_report(
        y_test.to_numpy(), predicted_categories, output_dict=True)

    df = pd.DataFrame(report).transpose()
    df.to_csv(f"{filename}.csv")


def balanced_classes(df, class_column):
    df_positives = df[df[class_column] == True]
    df_negatives = df[df[class_column] == False]

    df_positives = df_positives[:len(df_negatives.index)]
    df_balanced = pd.concat([df_positives, df_negatives])

    return df_balanced


def classify_with_tfidf():
    test_dataframe = preprocess_file("./Datasets/test.csv")
    train_dataframe = preprocess_file("./Datasets/train.csv")
    df = pd.concat([train_dataframe, test_dataframe])

    df_balanced = balanced_classes(df, "class")

    for classifier, name in zip(classifiers, classifiers_names):
        model = make_pipeline(TfidfVectorizer(), classifier)
        kf = KFold(n_splits=10, shuffle=True)
        i = 1

        for train_indicies, test_indicies in kf.split(df_balanced):
            x_train = df_balanced.iloc[train_indicies]["tweet"]
            x_test = df_balanced.iloc[test_indicies]["tweet"]
            y_train = df_balanced.iloc[train_indicies]["class"]
            y_test = df_balanced.iloc[test_indicies]["class"]

            filename = f"{name}_tfidf_{i}"
            i += 1
            classify(x_train=x_train, x_test=x_test, y_train=y_train,
                     y_test=y_test, model=model, filename=filename)

        joblib.dump(model, f"{name}_tfidf.sav")


def classify_with_custom_features():
    test_dataframe = preprocess_file("./Datasets/test.csv")
    train_dataframe = preprocess_file("./Datasets/train.csv")
    df = pd.concat([train_dataframe, test_dataframe])

    with open('./dict_inverted.json') as f:
        freq_inverted = json.load(f)
        f.close()

    df = feature_engineering(df, "tweet", freq_inverted)

    df_balanced = balanced_classes(df, "class")

    for classifier, name in zip(classifiers, classifiers_names):
        model = make_pipeline(classifier)
        kf = KFold(n_splits=10, shuffle=True)
        i = 1

        for train_indicies, test_indicies in kf.split(df_balanced):
            x_train = df_balanced.iloc[train_indicies][[
                'spoken', "rarity", "meanings", "lexical", "emoticon"]]
            x_test = df_balanced.iloc[test_indicies][[
                'spoken', "rarity", "meanings", "lexical", "emoticon"]]
            y_train = df_balanced.iloc[train_indicies]["class"]
            y_test = df_balanced.iloc[test_indicies]["class"]

            filename = f"{name}_cf_{i}"
            i += 1
            classify(x_train=x_train, x_test=x_test,
                     y_train=y_train, y_test=y_test, model=model, filename=filename)

        joblib.dump(model, f"{name}_cf.sav")


def main():
    classify_with_tfidf()
    classify_with_custom_features()


if __name__ == "__main__":
    main()
