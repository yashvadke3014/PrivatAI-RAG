"""
Step 4: the actual RAG query.

Usage:
    py query.py Legal "What is the notice period for termination?"
    py query.py Finance "What was Q3 revenue?"
    py query.py Legal "What was Q3 revenue?"      <-- isolation test!

Flow:
    question -> embed it -> search Qdrant WITH a team filter
             -> send only the retrieved chunks + question to Llama 3
             -> grounded answer
"""

import sys

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

COLLECTION_NAME = "enterprise_documents"

# The embedding model MUST be the same one used in ingest.py, otherwise the
# question's numbers and the stored chunks' numbers live in different "maps"
# and similarity search returns nonsense.
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# The LLM is your local Llama 3 served by Ollama. Without this line,
# LlamaIndex would default to OpenAI and ask for an API key.
Settings.llm = Ollama(model="llama3", request_timeout=180.0)


# The only teams that exist. In the real app this list will come from the
# user database, and the team will come from the logged-in user's account
# instead of being typed in.
VALID_TEAMS = {"legal": "Legal", "finance": "Finance"}


def normalize_team(team: str) -> str:
    """Map 'finance' / ' FINANCE ' to the stored tag 'Finance', or reject unknown teams."""
    key = team.strip().lower()
    if key not in VALID_TEAMS:
        raise ValueError(f"Unknown team '{team}'. Valid teams: {', '.join(VALID_TEAMS.values())}")
    return VALID_TEAMS[key]


def ask(team: str, question: str):
    team = normalize_team(team)
    client = QdrantClient(url="http://localhost:6333")
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    index = VectorStoreIndex.from_vector_store(vector_store)

    # THE multi-tenancy line: this filter is applied inside Qdrant, so chunks
    # from other teams are never even candidates for retrieval.
    team_filter = MetadataFilters(filters=[MetadataFilter(key="team", value=team)])

    query_engine = index.as_query_engine(filters=team_filter, similarity_top_k=3)
    response = query_engine.query(question)

    print(f"\n=== Asking as team: {team} ===")
    print(f"Question: {question}\n")
    if not response.source_nodes:
        print("No documents found for your team.\n")
        return
    print(f"Answer:\n{response}\n")

    print("--- Chunks the answer was built from ---")
    for node in response.source_nodes:
        print(f"  [team={node.metadata.get('team')}] (similarity {node.score:.2f}) "
              f"{node.metadata.get('file_name')}")

    # Safety net: fail loudly if anything from another team ever shows up.
    leaked = [n for n in response.source_nodes if n.metadata.get("team") != team]
    assert not leaked, "ISOLATION FAILURE: retrieved a chunk from another team!"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: py query.py <Team> "<question>"')
        sys.exit(1)
    try:
        ask(team=sys.argv[1], question=" ".join(sys.argv[2:]))
    except ValueError as e:
        print(e)
        sys.exit(1)