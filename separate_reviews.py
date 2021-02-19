import pandas as pd


def separate_labeled_and_unlabeled_reviews_to_files(filename_semi, filename_lbl, filename_unlbl, token):
    df = pd.read_csv(filename_semi)

    unlabeled_df = df[df['class'] == token]
    unlabeled_df = unlabeled_df[['comment', 'class']].copy()
    labeled_df = df[df['class'] != token]
    labeled_df = labeled_df[['comment', 'class']].copy()
    # print(labeled_df["class"].value_counts())
    # print(unlabeled_df["class"].value_counts())

    unlabeled_df.to_csv(filename_unlbl)
    labeled_df.to_csv(filename_lbl)


def main():
    separate_labeled_and_unlabeled_reviews_to_files(
        "reviews_semi_labeled.csv", "labeled_reviews.csv", "unlabeled_reviews.csv", "asd")


if __name__ == "__main__":
    main()
