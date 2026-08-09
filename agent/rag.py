"""
RAG knowledge source for the Researcher node.

Documents come from the Streamlit file_uploader at runtime,
so this module exposes index_documents() for app.py to call after upload
, and retrieve() for the Researcher node to call per task.
"""

from pathlib import Path

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

_retriever = None


def _read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def index_documents(file_paths: list[str], chunk_size: int = 200, chunk_overlap: int = 40) -> int:
    """
    Build a fresh in-memory index from the given file paths.

    Called by app.py once per new file_uploader batch. 
    Returns the number of chunks indexed, so the UI can show a confirmation.
    """
    global _retriever

    documents = [
        Document(text=_read_text(Path(p)), metadata={"source": Path(p).name})
        for p in file_paths
    ]

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)

    # Ephemeral (in-memory) client: each new upload batch gets a clean index.
    # there's no reason to persist across dashboard sessions —
    # a fresh upload should mean a fresh knowledge base, not an accumulating one.
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection("dashboard_kb")
    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=collection)
    )

    index = VectorStoreIndex(nodes, storage_context=storage_context)
    _retriever = index.as_retriever(similarity_top_k=4)
    return len(nodes)


def retrieve(query: str) -> tuple[str, list[str]]:
    """
    Return (context_text, sources). Returns ("", []) if nothing indexed yet 
    — the Researcher node must handle this explicitly, not crash.
    """
    if _retriever is None:
        return "", []
    hits = _retriever.retrieve(query)
    ctx = "\n".join(f"[{h.node.metadata.get('source', 'unknown')}] {h.node.text}" for h in hits)
    sources = sorted({h.node.metadata.get("source", "unknown") for h in hits})
    return ctx, sources


def has_index() -> bool:
    return _retriever is not None