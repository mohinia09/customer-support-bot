from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"
DEFAULT_CHROMA_DIR = Path(__file__).parent / "chroma_db"
DEFAULT_COLLECTION = "knowledge_base"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None


def _iter_txt_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*.txt") if p.is_file())


def _read_text_file(path: Path) -> str:
    # Use utf-8 with replacement to be robust to mixed encodings.
    return path.read_text(encoding="utf-8", errors="replace")


def _stable_chunk_id(source: str, chunk_index: int, text: str) -> str:
    h = hashlib.sha1()
    h.update(source.encode("utf-8"))
    h.update(b"\0")
    h.update(str(chunk_index).encode("utf-8"))
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _chunk_text(
    text: str,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Simple character-based chunking with overlap.
    Keeps implementation dependency-free (no LangChain required).
    """
    if not text:
        return []

    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - chunk_overlap
    return chunks


def _batch(iterable: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def _get_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in your environment or .env file.")
    return OpenAI(api_key=api_key)


def _get_chroma_collection(
    *,
    persist_dir: Path,
    collection_name: str,
) -> Any:
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(name=collection_name)


def build_vector_store(
    *,
    knowledge_base_dir: Path = DEFAULT_KNOWLEDGE_BASE_DIR,
    persist_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    force_recreate: bool = False,
) -> None:
    """
    Loads .txt files from knowledge_base_dir, embeds with OpenAI, and persists in ChromaDB.

    If force_recreate=True, the collection is deleted and rebuilt.
    Otherwise, ids are deterministic so re-running is safe (already-present ids are skipped).
    """
    collection = _get_chroma_collection(persist_dir=persist_dir, collection_name=collection_name)

    if force_recreate:
        client = chromadb.PersistentClient(path=str(persist_dir), settings=Settings(anonymized_telemetry=False))
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
        collection = client.get_or_create_collection(name=collection_name)

    files = list(_iter_txt_files(knowledge_base_dir))
    if not files:
        return

    openai_client = _get_openai_client()

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for file_path in files:
        raw = _read_text_file(file_path)
        chunks = _chunk_text(raw, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        source = str(file_path.relative_to(knowledge_base_dir)).replace("\\", "/")
        for idx, chunk in enumerate(chunks):
            chunk_id = _stable_chunk_id(source, idx, chunk)
            documents.append(chunk)
            metadatas.append({"source": source, "chunk_index": idx})
            ids.append(chunk_id)

    if not ids:
        return

    # Skip ids that already exist (fast path to make re-runs idempotent).
    existing: set[str] = set()
    for batch_ids in _batch(ids, 512):
        got = collection.get(ids=batch_ids, include=[])
        for existing_id in got.get("ids", []) or []:
            existing.add(existing_id)

    to_add_docs: List[str] = []
    to_add_metas: List[Dict[str, Any]] = []
    to_add_ids: List[str] = []

    for doc, meta, _id in zip(documents, metadatas, ids):
        if _id in existing:
            continue
        to_add_docs.append(doc)
        to_add_metas.append(meta)
        to_add_ids.append(_id)

    if not to_add_ids:
        return

    # Embed + add in batches.
    for docs_batch, metas_batch, ids_batch in zip(
        _batch(to_add_docs, 96),
        _batch(to_add_metas, 96),
        _batch(to_add_ids, 96),
    ):
        emb = openai_client.embeddings.create(model=embedding_model, input=docs_batch)
        vectors = [d.embedding for d in emb.data]
        collection.add(ids=ids_batch, documents=docs_batch, metadatas=metas_batch, embeddings=vectors)


def get_retriever(
    *,
    persist_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    k: int = 4,
) -> "callable[[str], List[RetrievedChunk]]":
    """
    Returns a function retrieve(query: str) -> List[RetrievedChunk]
    that performs semantic search over the local ChromaDB collection.
    """
    collection = _get_chroma_collection(persist_dir=persist_dir, collection_name=collection_name)
    openai_client = _get_openai_client()

    def retrieve(query: str) -> List[RetrievedChunk]:
        if not query.strip():
            return []
        q_emb = openai_client.embeddings.create(model=embedding_model, input=[query]).data[0].embedding
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        out: List[RetrievedChunk] = []
        for i in range(min(len(ids), len(docs), len(metas), len(dists))):
            out.append(
                RetrievedChunk(
                    id=str(ids[i]),
                    text=str(docs[i]),
                    metadata=dict(metas[i] or {}),
                    distance=float(dists[i]) if dists[i] is not None else None,
                )
            )
        return out

    return retrieve


if __name__ == "__main__":
    # Rebuild/update the local vector store from ./knowledge_base/*.txt
    build_vector_store()
