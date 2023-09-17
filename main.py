import concurrent.futures

from tqdm import tqdm

from data_processor import ArticleLoader, Text2Vec
from vector_db import VectorDB


def upload_vectors():
    print("Loading Data...")
    data = ArticleLoader.get_data()

    text2vec = Text2Vec()
    vdb = VectorDB(text2vec.vector_size)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for document in tqdm(data, desc="Vectorizing Data", total=len(data)):
            vector = text2vec.encode(document)
            executor.submit(vdb.upload_vector, vector, document)

    print("Done!")


def run_recommendation():
    data = """
We show that the anomaly of the positron fraction observed by the PAMELA
experiment can be attributed to recent supernova explosion(s) in a dense gas
cloud (DC) near the Earth. Protons are accelerated around the supernova remnant
(SNR). Electrons and positrons are created through hadronic interactions inside
the DC. Their spectrum is harder than that of the background because the SNR
spends much time in a radiative phase. Our scenario predicts that the
anti-proton flux dominates that of the background for >~100 GeV. We compare the
results with observations (Fermi, HESS, PPB-BETS, and ATIC).
        """

    text2vec = Text2Vec()
    vdb = VectorDB(text2vec.vector_size)
    vec = text2vec.encode(data)

    results = vdb.get_similar(vec)
    for i, result in enumerate(results['hits']['hits']):
        print(
            f"{i}:  Score - {result['_score']}\n{result['_source']['content']}\n\n----------------------------------------------------------\n\n")


if __name__ == '__main__':
    run_recommendation()
