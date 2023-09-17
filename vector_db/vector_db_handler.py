import hashlib
import os
from typing import List, Dict, Any

import yaml
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

import elastic_queries as queries


class VectorDB:
    def __init__(self, vector_dims: int):
        self.vector_dims = vector_dims
        self.db_conf = self._get_config("vector_db/db_config.yaml")
        self.queries_conf = self._get_config("vector_db/queries_config.yaml")
        self.client = self._get_client()
        self._create_index()

    def upload_vector(self, vector: List[float], content: str):
        _id = hashlib.sha256(content.encode('utf-8')).hexdigest()
        doc = {
            "vector": vector,
            "content": content
        }
        self.client.index(index=self.db_conf['index'], document=doc, id=_id)

    def get_similar(self, vector: List[float]) -> Dict[str, Any]:
        """
        use cosine similarity to retrieve vectors that are similar to the input vector
        """
        query = queries.similar_docs(self.queries_conf['min_score'], self.queries_conf['size'], vector)
        results = self.client.search(index=self.db_conf['index'], body=query)
        return results

    def clear_index(self):
        query = queries.all_docs()
        self.client.delete_by_query(index=self.db_conf['index'], body=query, conflicts="proceed")
        print("Index Cleared")

    def _create_index(self):
        mapping = queries.db_mapping(self.vector_dims)
        if not self.client.indices.exists(index=self.db_conf['index']):
            self.client.indices.create(index=self.db_conf['index'], body=mapping)

    def _get_client(self) -> Elasticsearch:
        if not os.path.exists(self.db_conf['cert_path']):
            raise FileNotFoundError("No HTTP Certificates Found")

        client = Elasticsearch(self.db_conf['host'],
                               ca_certs=self.db_conf['cert_path'],
                               basic_auth=("elastic", self.db_conf['password']))
        return client

    def document_exists(self, doc_id: str) -> bool:
        try:
            self.client.get(index=self.db_conf['index'], id=doc_id)
            return True
        except NotFoundError:
            return False

    @staticmethod
    def _get_config(path: str) -> Dict[str, Any]:
        with open(path, 'r') as file:
            config = yaml.load(file, yaml.FullLoader)
        return config
