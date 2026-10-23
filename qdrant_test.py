"""
Quick sanity test: prove that Qdrant's metadata filtering actually
enforces multi-tenant isolation before we build the real RAG pipeline
on top of it.

We insert a few fake "document chunks" tagged with a team, using tiny
made-up vectors (real embeddings come later via an embedding model —
right now we only care about proving the FILTERING mechanism works).
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "test_documents"

# Step 1: create a collection.
# size=4 is just for this toy test — real embeddings (BGE-M3 / MiniLM)
# will have much higher dimensions (e.g. 384 or 1024).
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=4, distance=Distance.COSINE),
)

# Step 2: insert fake chunks, each tagged with a "team" in its metadata.
# In the real project, this metadata gets attached automatically based
# on which team uploaded the document.
points = [
    PointStruct(
        id=1,
        vector=[0.1, 0.2, 0.3, 0.4],
        payload={"team": "Legal", "text": "This contract clause governs liability."},
    ),
    PointStruct(
        id=2,
        vector=[0.15, 0.25, 0.35, 0.45],
        payload={"team": "Legal", "text": "Termination requires 30 days notice."},
    ),
    PointStruct(
        id=3,
        vector=[0.9, 0.8, 0.7, 0.6],
        payload={"team": "Finance", "text": "Q3 revenue increased by 12%."},
    ),
    PointStruct(
        id=4,
        vector=[0.85, 0.75, 0.65, 0.55],
        payload={"team": "Finance", "text": "Budget forecast for next fiscal year."},
    ),
]
client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"Inserted {len(points)} test chunks across 2 teams.\n")

# Step 3: search WITHOUT any filter — a broken/insecure version, for comparison.
# This is what would happen if you forgot to filter by team — everyone sees everything.
print("=== UNFILTERED search (what NOT to do) ===")
unfiltered_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=[0.1, 0.2, 0.3, 0.4],  # searching as if we're a Legal user
    limit=4,
).points
for r in unfiltered_results:
    print(f"  team={r.payload['team']:8s} | {r.payload['text']}")

# Step 4: search WITH a metadata filter — this is the actual multi-tenancy mechanism.
# A "Legal" user's query should ONLY ever be able to retrieve Legal chunks,
# no matter how similar a Finance chunk's vector might be.
print("\n=== FILTERED search (team='Legal' only) ===")
filtered_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=[0.1, 0.2, 0.3, 0.4],
    query_filter=Filter(
        must=[FieldCondition(key="team", match=MatchValue(value="Legal"))]
    ),
    limit=4,
).points
for r in filtered_results:
    print(f"  team={r.payload['team']:8s} | {r.payload['text']}")

# Sanity assertion: prove NO Finance data leaked into the Legal-filtered results.
leaked = [r for r in filtered_results if r.payload["team"] != "Legal"]
assert not leaked, "ISOLATION FAILURE: Finance data leaked into Legal's results!"
print("\n✅ Isolation confirmed: filtered search returned ONLY Legal team data.")