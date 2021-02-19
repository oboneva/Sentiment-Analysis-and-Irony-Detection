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

classifiers = [MultinomialNB(), SVC(probability=True), RandomForestClassifier(),
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
    df = df.dropna()
    modify_classes(df)

    return df


def merge_data():
    test_dataframe = preprocess_file("./Datasets/test.csv")
    test_dataframe.to_csv("./Datasets/test_cleaned.csv")

    train_dataframe = preprocess_file("./Datasets/train.csv")
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


def classify_tfidf_all(df):
    for classifier, name in zip(classifiers, classifiers_names):
        model = make_pipeline(TfidfVectorizer(), classifier)
        kf = KFold(n_splits=3, shuffle=True)
        i = 1

        for train_indicies, test_indicies in kf.split(df):
            x_train = df.iloc[train_indicies]["comment"]
            x_test = df.iloc[test_indicies]["comment"]
            y_train = df.iloc[train_indicies]["class"]
            y_test = df.iloc[test_indicies]["class"]

            filename = f"{name}_tfidf_all_{i}"
            i += 1
            classify(x_train=x_train, x_test=x_test, y_train=y_train,
                     y_test=y_test, model=model, filename=filename)

        joblib.dump(model, f"{name}_tfidf_all.sav")


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


def balanced_reviews_tweets(positives, negatives):
    tweets_df = preprocess_file("./Datasets/test.csv")
    tweets_df = tweets_df.rename(columns={"tweet": "comment"})
    reviews_df = pd.read_csv("./labeled_reviews.csv")
    reviews_df = reviews_df[['comment', 'class']]

    tweets_df_positives = tweets_df[tweets_df["class"] == True]
    tweets_df_negatives = tweets_df[tweets_df["class"] == False]

    tweets_df_positives = tweets_df_positives[:positives]
    tweets_df_positives = tweets_df_positives[['comment', 'class']]
    tweets_df_negatives = tweets_df_negatives[:negatives]
    tweets_df_negatives = tweets_df_negatives[['comment', 'class']]

    result = pd.concat([tweets_df_positives, tweets_df_negatives, reviews_df])

    return result


def train_test_sets(df, k):
    test = df[: len(df) // k]
    train = df[len(df) // k:]

    return train, test


def train_test_balanced_reviews_tweets(positives, negatives):
    tweets_df = preprocess_file("./Datasets/test.csv")
    tweets_df = tweets_df.rename(columns={"tweet": "comment"})
    reviews_df = pd.read_csv("./labeled_reviews.csv")
    reviews_df = reviews_df[['comment', 'class']]

    tweets_df_positives = tweets_df[tweets_df["class"] == True]
    tweets_df_negatives = tweets_df[tweets_df["class"] == False]
    tweets_df_positives = tweets_df_positives[:positives]
    tweets_df_positives = tweets_df_positives[['comment', 'class']]
    tweets_df_negatives = tweets_df_negatives[:negatives]
    tweets_df_negatives = tweets_df_negatives[['comment', 'class']]

    train_reviews_df, test_reviews_df = train_test_sets(reviews_df, 3)
    tweets_df_positives_train, tweets_df_positives_test = train_test_sets(
        tweets_df_positives, 3)
    tweets_df_negatives_train, tweets_df_negatives_test = train_test_sets(
        tweets_df_negatives, 3)

    train = pd.concat([tweets_df_positives_train,
                       tweets_df_negatives_train, train_reviews_df])

    test = pd.concat([tweets_df_positives_test,
                      tweets_df_negatives_test, test_reviews_df])

    return train, test


def semi_sup():
    df_labeled_train, df_labeled_test = train_test_balanced_reviews_tweets(
        1000, 1000)

    df_unlabeled = pd.read_csv("unlabeled_reviews.csv")
    model = joblib.load("./svm_tfidf_all.sav")

    high_prob = [1]
    while len(high_prob) > 0:
        model.fit(df_labeled_train["comment"].to_numpy(),
                  df_labeled_train["class"].to_numpy())
        predicted_categories = model.predict(
            df_unlabeled["comment"].to_numpy())
        predicted_categories_proba = model.predict_proba(
            df_unlabeled["comment"].to_numpy())

        prob_false = predicted_categories_proba[:, 0]
        prob_true = predicted_categories_proba[:, 1]

        df_prob = pd.DataFrame([])
        df_prob['preds'] = predicted_categories
        df_prob['prob_false'] = prob_false
        df_prob['prob_true'] = prob_true
        df_prob.index = df_unlabeled.index

        high_prob = pd.concat([df_prob.loc[df_prob['prob_false'] > 0.99],
                               df_prob.loc[df_prob['prob_true'] > 0.99]],
                              axis=0)

        pseudos = df_unlabeled.loc[high_prob.index]
        pseudos["class"] = high_prob['preds']
        df_labeled_train = pd.concat(
            [df_labeled_train, pseudos[['comment', 'class']]], axis=0)
        df_unlabeled = df_unlabeled.drop(index=high_prob.index)

        test = model.predict(df_labeled_test["comment"].to_numpy())
        report = classification_report(
            df_labeled_test["class"].to_numpy(), test)


def main():
    # classify_with_tfidf()
    # classify_with_custom_features()
    semi_sup()


if __name__ == "__main__":
    main()
