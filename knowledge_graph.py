import re

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_knowledge_graph(documents, threshold=0.15):
    """Connect similar sentences extracted from PDF documents.

    Args:
        documents: Mapping of document names to extracted text.
        threshold: Minimum cosine similarity required for an edge.

    Returns:
        A NetworkX graph whose nodes represent sentences and edges represent similarity.
    """
    graph = nx.Graph()
    sentences = []
    sources = []

    for filename, text in documents.items():
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = sentence.strip()
            if len(sentence.split()) >= 4:
                sentences.append(sentence)
                sources.append(filename)

    if not sentences:
        return graph

    try:
        vectors = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    except ValueError:
        return graph

    similarities = cosine_similarity(vectors)
    for i, sentence in enumerate(sentences):
        graph.add_node(
            f"topic_{i}", label=sentence[:65], source=sources[i], details=sentence
        )

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            score = float(similarities[i, j])
            if score >= threshold:
                graph.add_edge(f"topic_{i}", f"topic_{j}", weight=score)

    return graph
