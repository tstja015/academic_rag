# diagnostic.py
import chromadb
import config
from sentence_transformers import SentenceTransformer, CrossEncoder

# Load models
client = chromadb.PersistentClient(path=config.DB_DIR)
collection = client.get_collection(config.COLLECTION_NAME)

embedder = SentenceTransformer(config.EMBED_MODEL)
reranker = CrossEncoder(config.RERANK_MODEL)

# The query that failed
query = "Tell me about MCSA within context of CPD protocols"

# Step 1: Embedding retrieval from ChromaDB
query_vec = embedder.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_vec],
    n_results=10,
    include=["documents", "metadatas", "distances"],
)

print("=" * 60)
print("STEP 1: ChromaDB retrieval (cosine similarity)")
print("=" * 60)
for i in range(len(results["documents"][0])):
    meta = results["metadatas"][0][i]
    dist = results["distances"][0][i]
    print(f"\n--- Result {i+1} ---")
    print(f"  filename: {meta.get('filename', '?')}")
    print(f"  section:  {meta.get('section', '?')}")
    print(f"  cosine:   {1.0 - dist:.3f}")
    print(f"  doc[:120]: {results['documents'][0][i][:120]}")

# Step 2: Reranker scoring
print("\n" + "=" * 60)
print("STEP 2: Cross-encoder rerank scores")
print(f"Current RERANK_THRESHOLD = {config.RERANK_THRESHOLD}")
print("=" * 60)

docs = results["documents"][0]
pairs = [(query, doc) for doc in docs]
scores = reranker.predict(pairs)

for i, (score, doc) in enumerate(zip(scores, docs)):
    meta = results["metadatas"][0][i]
    status = "PASS" if score >= config.RERANK_THRESHOLD else "FILTERED OUT"
    print(f"\n  [{status}] Score: {score:.3f}  |  {meta.get('filename', '?')}")
    print(f"            {doc[:100]}...")

# Summary
passing = sum(1 for s in scores if s >= config.RERANK_THRESHOLD)
print("\n" + "=" * 60)
print(f"SUMMARY: {passing}/{len(scores)} chunks pass threshold of {config.RERANK_THRESHOLD}")
if passing == 0:
    print(">>> ALL CHUNKS FILTERED OUT -- this is why you get general knowledge fallback")
    print(f">>> Best score: {max(scores):.3f}")
    print(f">>> Suggestion: set RERANK_THRESHOLD = {min(-2.0, min(scores) - 0.5):.1f} in config.py")
print("=" * 60)
