"""RAG engine: index files into Qdrant and answer questions, always scoped to ONE team.

Heavy libraries (LlamaIndex, torch, the embedding model) are imported lazily,
inside functions, so the API starts instantly and tests can run without them.
"""
from functools import lru_cache
from pathlib import Path

from . import config

NO_DOCS = "No documents found for your team yet."

QA_PROMPT = (
    "You are an internal company assistant. Answer the question using ONLY the "
    "context below. If the context does not contain the answer, reply exactly: "
    "\"I couldn't find that in your team's documents.\"\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Answer: "
)


@lru_cache(maxsize=1)
def _client():
    """One-time setup: load the embedding model, connect the LLM, connect Qdrant."""
    from llama_index.core import Settings
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.ollama import Ollama
    from qdrant_client import QdrantClient

    Settings.embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL)
    Settings.llm = Ollama(model=config.OLLAMA_MODEL, request_timeout=180.0)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    return QdrantClient(url=config.QDRANT_URL)


def _vector_store():
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    return QdrantVectorStore(client=_client(), collection_name=config.COLLECTION_NAME)


def ingest_file(path: Path, team: str, uploaded_by: str, display_name: str) -> None:
    """Chunk + embed one file and store it in Qdrant tagged with `team`."""
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex

    store = _vector_store()
    documents = SimpleDirectoryReader(input_files=[str(path)]).load_data()
    for doc in documents:
        # Replace the default metadata (which includes the server file path)
        # with only what we need. 'team' is the isolation tag.
        doc.metadata = {"team": team, "uploaded_by": uploaded_by, "file_name": display_name}
        # Keep metadata out of the text that gets embedded / shown to the LLM,
        # so it's only used for filtering and display.
        doc.excluded_embed_metadata_keys = list(doc.metadata)
        doc.excluded_llm_metadata_keys = list(doc.metadata)
    VectorStoreIndex.from_documents(
        documents, storage_context=StorageContext.from_defaults(vector_store=store)
    )


def ask(team: str, question: str) -> dict:
    """Answer `question` using only `team`'s documents."""
    from llama_index.core import PromptTemplate, VectorStoreIndex
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

    client = _client()
    if not client.collection_exists(config.COLLECTION_NAME):
        return {"answer": NO_DOCS, "sources": []}

    index = VectorStoreIndex.from_vector_store(_vector_store())
    # THE multi-tenancy line: applied inside Qdrant, so other teams' chunks
    # are never even candidates.
    team_filter = MetadataFilters(filters=[MetadataFilter(key="team", value=team)])
    engine = index.as_query_engine(
        filters=team_filter,
        similarity_top_k=3,
        text_qa_template=PromptTemplate(QA_PROMPT),
    )
    response = engine.query(question)

    nodes = response.source_nodes
    if not nodes:
        return {"answer": NO_DOCS, "sources": []}

    # Safety net: fail loudly if a chunk from another team ever appears.
    if any(n.metadata.get("team") != team for n in nodes):
        raise RuntimeError("ISOLATION FAILURE: retrieved a chunk from another team")

    sources, seen = [], set()
    for n in nodes:
        name = n.metadata.get("file_name")
        if name not in seen:
            seen.add(name)
            sources.append({"file": name, "score": round(float(n.score or 0), 2)})
    return {"answer": str(response).strip(), "sources": sources}