import pandas as pd


def clean_reviews(filename_from, filename_to):
    reviews_df = pd.read_csv(filename_from)
    reviews_df["comment"] = reviews_df["comment"].apply(lambda x: x.strip())
    reviews_df.to_csv(filename_to)


def main():
    clean_reviews("./Datasets/reviews_raw.csv", "./Datasets/reviews.csv")


if __name__ == "__main__":
    main()
