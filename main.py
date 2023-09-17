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
Crohn's disease (CD) is associated with an ecological imbalance of the
intestinal microbiota, consisting of hundreds of species. The underlying
complexity as well as individual differences between patients contributes to
the difficulty to define a standardized treatment. Computational modeling can
systematically investigate metabolic interactions between gut microbes to
unravel novel mechanistic insights. In this study, we integrated metagenomic
data of CD patients and healthy controls with genome-scale metabolic models
into personalized in silico microbiotas. We predicted short chain fatty acid
(SFCA) levels for patients and controls, which were overall congruent with
experimental findings. As an emergent property, low concentrations of SCFA were
predicted for CD patients and the SCFA signatures were unique to each patient.
Consequently, we suggest personalized dietary treatments that could improve
each patient's SCFA levels. The underlying modeling approach could aid clinical
practice to find novel dietary treatment and guide recovery by rationally
proposing food aliments.
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
