import json
import os
import pickle
from typing import List

import yaml
from tqdm import tqdm

from .loader import Loader


class ArticleLoader(Loader):
    with open("data_processor/config.yaml", 'r') as file:
        config = yaml.load(file, yaml.FullLoader)

    @classmethod
    def get_data(cls) -> List[str]:
        if not os.path.exists(cls.config["processed_path"]):
            cls._prepare_data()

        with open(cls.config["processed_path"], 'rb') as file:
            articles = pickle.load(file)
        return articles

    @classmethod
    def _prepare_data(cls):
        """
        removes the unnecessary data from the original file of metadata about articles
        saves a pkl file with a list of articles' abstract content
        """
        articles_metadata = cls._load_data()
        articles = cls.extract_abstract(articles_metadata)
        print("Saving Data")
        with open(cls.config["processed_path"], 'wb') as file:
            pickle.dump(articles, file)
        print("Done")

    @classmethod
    def _load_data(cls) -> List[str]:
        """
        gets raw data as a large string and separate it to strings containing articles metadata in json format
        """
        print("Reading Raw Data")
        with open(cls.config["raw_data_path"], 'r') as file:
            raw_data = file.read()
        return raw_data.split('\n')

    @staticmethod
    def extract_abstract(articles_metadata: List[str]) -> List[str]:
        articles = []
        for article in tqdm(articles_metadata, desc="Preparing Data", total=len(articles_metadata)):
            try:
                abstract = json.loads(article)['abstract']
                articles.append(abstract)
            except json.decoder.JSONDecodeError:
                print("JSON Parse Failed")
        return articles
