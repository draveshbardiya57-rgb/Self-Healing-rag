# hybrid_retriever.py
# ============================================================
# Domain-aware retrieval strategy
# Developer docs  → vector only (BGE sufficient)
# Legal/Finance/Health → hybrid (BM25 + vector)
# ============================================================
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from dotenv import load_dotenv
import hashlib

load_dotenv()

# ── LAZY EMBEDDINGS ─────────────────────────────────────────
# Deferred until first call so importing this module does NOT
# trigger CUDA initialisation (fixes segfault with DeepEval).
_EMBEDDINGS_CACHE = None

def _get_embeddings():
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is None:
        device = "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES") == "" else "cuda"
        _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _EMBEDDINGS_CACHE

CHROMA_BASE = "./chroma_db"

# Cache — build retriever once per domain per session
_retriever_cache = {}

# Domain strategy config
# Tune weights per domain based on keyword density
DOMAIN_CONFIG = {
    "developer_docs": {
        "use_hybrid": True,
        "weights":    [0.7, 0.3],  # favor semantic for code docs
        "reason":     "BGE handles Python APIs well — light BM25 boost"
    },
    "legal": {
        "use_hybrid": True,
        "weights":    [0.4, 0.6],  # favor BM25 for exact legal refs
        "reason":     "Section numbers, case IDs need exact matching"
    },
    "finance": {
        "use_hybrid": True,
        "weights":    [0.4, 0.6],  # favor BM25 for ticker/reg numbers
        "reason":     "Regulation numbers, circular IDs need exact matching"
    },
    "health": {
        "use_hybrid": True,
        "weights":    [0.5, 0.5],  # balanced for medical terminology
        "reason":     "Drug names and ICD codes need both approaches"
    },
}


def load_all_documents(domain: str) -> list[Document]:
    """
    Loads all chunks from ChromaDB for BM25 indexing.
    Paginated to avoid SQLite's max-variables limit on large collections —
    without this, .get() on the whole collection fails once chunk count
    grows past a few thousand, depending on ChromaDB's internal batching.
    """
    vectorstore = Chroma(
        persist_directory=f"{CHROMA_BASE}/{domain}",
        embedding_function=_get_embeddings(),
        collection_name=domain
    )

    print(f"  Loading documents for BM25 index (paginated)...")

    documents = []
    batch_size = 500
    offset = 0

    while True:
        result = vectorstore.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset
        )

        batch_docs = result["documents"]
        if not batch_docs:
            break

        for content, metadata in zip(batch_docs, result["metadatas"]):
            documents.append(Document(
                page_content=content,
                metadata=metadata or {}
            ))

        offset += batch_size
        print(f"    Loaded {len(documents)} so far...")

    print(f"  Indexed {len(documents)} documents into BM25")
    return documents


def build_retriever(domain: str, k: int = 4):
    """
    Builds the appropriate retriever for the domain.
    Always hybrid — weights tuned per domain.
    """
    config = DOMAIN_CONFIG.get(domain, DOMAIN_CONFIG["developer_docs"])

    print(f"\n[RETRIEVER] Domain: {domain}")
    print(f"  Strategy: Hybrid (vector + BM25)")
    print(f"  Weights:  {config['weights']} — {config['reason']}")

    # Vector retriever
    vectorstore = Chroma(
        persist_directory=f"{CHROMA_BASE}/{domain}",
        embedding_function=_get_embeddings(),
        collection_name=domain
    )
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    # BM25 retriever
    all_docs = load_all_documents(domain)
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k

    # Combine with domain-specific weights
    hybrid = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=config["weights"]
    )

    print(f"  ✅ Hybrid retriever ready")
    return hybrid


def get_retriever(domain: str, k: int = 4):
    """Returns cached retriever — builds once per session."""
    cache_key = f"{domain}_{k}"
    if cache_key not in _retriever_cache:
        _retriever_cache[cache_key] = build_retriever(domain, k)
    return _retriever_cache[cache_key]


def hybrid_search(
    query:  str,
    domain: str,
    k:      int = 4
) -> list[Document]:
    """
    Main search function used by graph.py Node 1.
    Always uses hybrid retrieval with domain-tuned weights.
    """
    retriever = get_retriever(domain, k)
    documents = retriever.invoke(query)

    # Deduplicate using content hash
    seen   = set()
    unique = []
    for doc in documents:
        key = hashlib.md5(doc.page_content.encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(doc)

    return unique[:k]