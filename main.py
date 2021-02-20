import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import seaborn as sns
import json
import joblib

from tweet_preprocess import clean
from feature_engineering import feature_engineering
from frequency_dict import create_inverted_frequency_dict

sns.set()

classifiers = [MultinomialNB(), SVC(probability=True), RandomForestClassifier(),
               KNeighborsClassifier(n_neighbors=3), DecisionTreeClassifier()]
classifiers_names = ["naive_bayes", "svm", "random_forest",
                     "knn3", "decision_tree"]


def modify_classes(df):
    df["class"] = df["class"].replace(['figurative', 'irony', 'sarcasm'], True)
    df["class"] = df["class"].replace(['regular'], False)


def preprocess_file(filepath):
    df = pd.read_csv(filepath)
    df = df.rename(columns={"tweets": "tweet"})
    df = clean(df, "tweet")
    df = df.dropna()
    modify_classes(df)

    return df


def plot(report, filename, label):
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
    plt.yticks(rotation=0)
    plt.title(label)
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()


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


def semi_sup_cf():
    # df_review = pd.read_csv("./Datasets/labeled_reviews.csv")
    # df_tweet = preprocess_file("./Datasets/test.csv")
    # df_tweet = df_tweet.rename(columns={"tweet": "comment"})
    # df_review = df_review[['comment', 'class']]
    # df = pd.concat([df_tweet, df_review])

    # freq_inverted = create_inverted_frequency_dict(df, "comment")
    # json_dict = json.dumps(freq_inverted)
    # f = open("dict_inverted.json", "w")
    # f.write(json_dict)
    # f.close()

    with open('./dict_inverted.json') as f:
        freq_inverted = json.load(f)
        f.close()

    for classifier, name in zip(classifiers, classifiers_names):
        model = make_pipeline(classifier)

        df_labeled_train, df_labeled_test = train_test_balanced_reviews_tweets(
            1000, 1000)
        df_labeled_train = feature_engineering(
            df_labeled_train, "comment", freq_inverted)
        df_labeled_test = feature_engineering(
            df_labeled_test, "comment", freq_inverted)

        df_unlabeled = pd.read_csv("unlabeled_reviews.csv")
        df_unlabeled = feature_engineering(
            df_unlabeled, "comment", freq_inverted)

        high_prob = [1]
        i = 0
        while True:
            model.fit(df_labeled_train[[
                'spoken', "rarity", "meanings", "lexical", "emoticon"]].to_numpy(),
                df_labeled_train["class"].to_numpy())
            predicted_categories = model.predict(
                df_unlabeled[[
                    'spoken', "rarity", "meanings", "lexical", "emoticon"]].to_numpy())
            predicted_categories_prob = model.predict_proba(
                df_unlabeled[[
                    'spoken', "rarity", "meanings", "lexical", "emoticon"]].to_numpy())

            prob_false = predicted_categories_prob[:, 0]
            prob_true = predicted_categories_prob[:, 1]

            df_prob = pd.DataFrame([])
            df_prob['predicted'] = predicted_categories
            df_prob['prob_false'] = prob_false
            df_prob['prob_true'] = prob_true
            df_prob.index = df_unlabeled.index

            high_prob = pd.concat([df_prob.loc[df_prob['prob_false'] > 0.99],
                                   df_prob.loc[df_prob['prob_true'] > 0.99]], axis=0)

            pseudos = df_unlabeled.loc[high_prob.index]
            pseudos["class"] = high_prob['predicted']
            df_labeled_train = pd.concat(
                [df_labeled_train, pseudos[[
                    'spoken', "rarity", "meanings", "lexical", "emoticon", 'class']]], axis=0)
            df_unlabeled = df_unlabeled.drop(index=high_prob.index)

            if len(df_unlabeled) == 0 or len(high_prob) == 0:
                test = model.predict(df_labeled_test[[
                    'spoken', "rarity", "meanings", "lexical", "emoticon"]].to_numpy())
                report = classification_report(
                    df_labeled_test["class"].to_numpy(), test, output_dict=True)
                plot(report, f"{name}_cf.png", f"{name} with custom features")
                joblib.dump(model, f"{name}_cf.sav")
                break

            i += 1


def semi_sup_tfidf():
    for classifier, name in zip(classifiers, classifiers_names):
        model = make_pipeline(TfidfVectorizer(), classifier)

        df_labeled_train, df_labeled_test = train_test_balanced_reviews_tweets(
            1000, 1000)

        df_unlabeled = pd.read_csv("unlabeled_reviews.csv")

        high_prob = [1]
        i = 0
        while True:
            model.fit(df_labeled_train["comment"].to_numpy(),
                      df_labeled_train["class"].to_numpy())
            predicted_categories = model.predict(
                df_unlabeled["comment"].to_numpy())
            predicted_categories_prob = model.predict_proba(
                df_unlabeled["comment"].to_numpy())

            prob_false = predicted_categories_prob[:, 0]
            prob_true = predicted_categories_prob[:, 1]

            df_prob = pd.DataFrame([])
            df_prob['predicted'] = predicted_categories
            df_prob['prob_false'] = prob_false
            df_prob['prob_true'] = prob_true
            df_prob.index = df_unlabeled.index

            high_prob = pd.concat([df_prob.loc[df_prob['prob_false'] > 0.99],
                                   df_prob.loc[df_prob['prob_true'] > 0.99]], axis=0)

            pseudos = df_unlabeled.loc[high_prob.index]
            pseudos["class"] = high_prob['predicted']
            df_labeled_train = pd.concat(
                [df_labeled_train, pseudos[['comment', 'class']]], axis=0)
            df_unlabeled = df_unlabeled.drop(index=high_prob.index)

            if len(df_unlabeled) == 0 or len(high_prob) == 0:
                test = model.predict(df_labeled_test["comment"].to_numpy())
                report = classification_report(
                    df_labeled_test["class"].to_numpy(), test, output_dict=True)
                plot(report, f"{name}_tfidf.png", f"{name} with tfidf")
                joblib.dump(model, f"{name}_tfidf.sav")
                break

            i += 1


def main():
    semi_sup_tfidf()
    semi_sup_cf()


if __name__ == "__main__":
    main()
