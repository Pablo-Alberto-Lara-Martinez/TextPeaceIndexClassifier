# src/embeddings.py
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
import uuid

def compute_embeddings_chroma(texts, embedding_type="chroma", batch_size=5000):
    """
    Calculates embeddings using ChromaDB for a list of texts in batches.
    Returns a numpy array of shape (n_texts, dim).
    """
    client = chromadb.Client()
    
    if embedding_type == "chroma":
        default_ef = embedding_functions.DefaultEmbeddingFunction()
    else:
        # Doc2Vec vector
        d2v = embedding_functions.Doc2VecEmbeddingFunction(model_name="dbow300")
        default_ef = d2v
        
    # Ephemeral collection with a unique name to avoid ID collisions
    collection = client.get_or_create_collection(
        name=f"embeds_{uuid.uuid4().hex[:8]}",
        embedding_function=default_ef
    )
    
    ids = [f"doc-{i}" for i in range(len(texts))]
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        
        collection.add(documents=batch_texts, ids=batch_ids)
        batch_embeddings = collection.get(ids=batch_ids, include=["embeddings"])['embeddings']
        embeddings.extend(batch_embeddings)

    return np.array(embeddings, dtype=np.float32)

def prepare_country_level(df):
    """
    Processes a dataframe with multiple texts per country:
    - Calculates embeddings per text.
    - Averages embeddings per country.
    - Determines peace status per country (majority of labels).
    
    Returns: lists of countries, X_country, y_country, and grouped indices.
    """
    from .data_utils import majority_label
    
    texts = df["text"].tolist()
    E = compute_embeddings_chroma(texts) 
    
    # Group indices by country
    groups = {}
    for i, country in enumerate(df["country"]):
        groups.setdefault(country, []).append(i)
        
    countries = sorted(groups.keys()) # Stable/alphabetical order
    X_country, y_country = [], []
    
    for country in countries:
        idxs = groups[country]
        # Average embeddings for the texts
        emb_mean = E[idxs].mean(axis=0)
        X_country.append(emb_mean)
        
        # Majority label
        labels = df.iloc[idxs]["peace"].astype(int).values
        y_country.append(majority_label(labels))

    return countries, np.array(X_country), np.array(y_country, dtype=int), groups