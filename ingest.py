"""
Step 3: Real document ingestion pipeline.

Takes a text file, chunks it, embeds each chunk with a LOCAL embedding
model (no API key, no internet call needed once downloaded), tags each
chunk with which team it belongs to, and stores it in Qdrant.

This replaces the hand-typed fake vectors from qdrant_test.py with a
real pipeline on real document content.
"""

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

COLLECTION_NAME = "enterprise_documents"

# Use a small, fast, local embedding model — runs on CPU fine, no API
# key needed. This downloads once (~90MB) and is cached afterward.
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Split documents into chunks of ~512 tokens with some overlap, so
# context isn't abruptly cut off at chunk boundaries.
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

client = QdrantClient(url="http://localhost:6333")


def reset_collection():
    """Delete the collection if it exists so re-running never creates duplicates.
    (Fine for test data. The real app will skip already-ingested files instead.)"""
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted old collection '{COLLECTION_NAME}'")


# Runs at import time on purpose: it must happen BEFORE the vector store
# object is created below. Don't import this file from other scripts.
reset_collection()
vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


def ingest_document(file_path: str, team: str):
    """Load one document, chunk + embed it, and store it tagged with `team`."""
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    # Tag every document (and therefore every chunk derived from it)
    # with the team metadata. This is the tag that Qdrant's filter
    # will check later to enforce isolation.
    for doc in documents:
        doc.metadata["team"] = team

    VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    print(f"Ingested '{file_path}' tagged as team='{team}' ({len(documents)} doc(s))")


if __name__ == "__main__":
    ingest_document("sample_docs/legal_contract_notes.txt", team="Legal")
    ingest_document("sample_docs/finance_q3_report.txt", team="Finance")
    print("\nIngestion complete. Check the Qdrant dashboard to see the stored chunks:")
    print("http://localhost:6333/dashboard")