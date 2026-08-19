import faiss
import numpy as np
import pandas as pd


EMBEDDING_PATH = "data/processed/article_embeddings.npy"
ARTICLE_IDS_PATH = "data/processed/article_ids.csv"
INDEX_PATH = "data/processed/article_index.faiss"


embeddings = np.load(EMBEDDING_PATH).astype("float32")
article_ids = pd.read_csv(ARTICLE_IDS_PATH)

print("Embeddings:", embeddings.shape)
print("Articles:", article_ids.shape)

dimension = embeddings.shape[1]

# Inner product = cosine similarity because
# our article embeddings are already normalized
index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

faiss.write_index(
    index,
    INDEX_PATH
)

print("FAISS index size:", index.ntotal)
print("Saved to:", INDEX_PATH)