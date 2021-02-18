from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pandas as pd


es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

keys = ['stars', 'comment']


def filter_keys(document):
    return {key: document[key] for key in keys}


def doc_generator(df):
    df_iter = df.iterrows()
    for index, document in df_iter:
        yield {
            "_index": 'sarcastic_reviews_index',
            "_type": "_doc",
            "_id": f"{index}",
            "_source": filter_keys(document),
        }


def serach_tokens(index, tokens):
    res = es.search(index=index,
                    body={"query": {"match": {"comment": tokens}}})

    return res['hits']['hits']


def search_phrase(index, phrase):
    res = es.search(index=index, body={
                    "query": {"match_phrase": {"comment": phrase}}})

    return res['hits']['hits']


def main():
    # reviews_df = pd.read_csv("reviews.csv")
    # helpers.bulk(es, doc_generator(reviews_df))

    # for hit in res['hits']['hits']:
    #     print(hit)


if __name__ == "__main__":
    main()
