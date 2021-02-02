import pandas as pd


def clean_reviews():
    reviews_df = pd.read_csv("reviews_raw.csv")
    reviews_df["comment"] = reviews_df["comment"].apply(lambda x: x.strip())
    reviews_df.to_csv("reviews.csv")


def main():
    clean_reviews()


if __name__ == "__main__":
    main()
