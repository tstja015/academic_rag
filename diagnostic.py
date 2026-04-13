import chromadb
import config
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path=config.DB_DIR)
collection = client.get_collection(config.COLLECTION_NAME)

embedder = SentenceTransformer(config.EMBED_MODEL)
query_vec = embedder.encode("fish fluoxetine").tolist()

results = collection.query(
    query_embeddings=[query_vec],
    n_results=3,
    include=["documents", "metadatas", "distances"],
)

print("Top-level keys:", list(results.keys()))
print()

for i in range(len(results["documents"][0])):
    print(f"--- Result {i+1} ---")
    print("  metadata:", results["metadatas"][0][i])
    print("  distance:", results["distances"][0][i])
    print("  doc[:100]:", results["documents"][0][i][:100])
    print()
