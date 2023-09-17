from typing import Dict, Any, List


def similar_docs(min_score: float, size: int, vector: List[float]) -> Dict[str, Any]:
    """
    :param min_score: return documents with higher similarity. between 1 (least similar) and 2 (most similar)
    :param size: how many documents to retrieve (given they have higher score than min_score)
    :param vector: the vector to compare with
    """
    return {
        "min_score": min_score,
        "size": size,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                    "params": {"query_vector": vector}
                }
            }
        }
    }


def all_docs():
    return {
        "query": {
            "match_all": {}
        }
    }


def db_mapping(vector_dims: int):
    return {
        "mappings": {
            "properties": {
                "vector": {
                    "type": "dense_vector",
                    "dims": vector_dims
                },
                "content": {
                    "type": "text"
                }
            }
        }
    }
