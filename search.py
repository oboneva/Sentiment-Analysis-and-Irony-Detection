from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pandas as pd


es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

use_these_keys = ['stars', 'comment']


def filterKeys(document):
    return {key: document[key] for key in use_these_keys}


def doc_generator(df):
    df_iter = df.iterrows()
    for index, document in df_iter:
        yield {
            "_index": 'sarcastic_reviews_index',
            "_type": "_doc",
            "_id": f"{index}",
            "_source": filterKeys(document),
        }


def main():
    reviews_df = pd.read_csv("reviews.csv")
    helpers.bulk(es, doc_generator(reviews_df))

    res = es.search(index="sarcastic_reviews_index",
                    body={"query": {"match_all": {}}})

    # for hit in res['hits']['hits']:
    #     print(hit)


if __name__ == "__main__":
    main()
