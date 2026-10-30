"""
Central configuration for the Advanced RAG project.

Every tunable value (API keys, model names, chunk sizes, how many documents to
retrieve) lives here so that the rest of the code never has to hard-code them.
If you want to experiment with the pipeline, this is the first file to edit.

Values are read from a `.env` file (see `.env.example`), with sensible
defaults so the project runs even if you only set `OPENAI_API_KEY`.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Read the .env file once, when this module is first imported.
# After this call, os.getenv("OPENAI_API_KEY") returns the value from .env.
load_dotenv()


def _get_int(name: str, default: int) -> int:
    """
    Read an integer setting from the environment.

    Everything in a `.env` file is text, so "400" arrives as the string "400".
    This helper converts it to a real integer and falls back to the default if
    the variable is missing or is not a valid number.

    Args:
        name: Environment variable name, for example "CHILD_CHUNK_SIZE".
        default: Value to use when the variable is missing or invalid.

    Returns:
        The setting as an integer.
    """
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError:
        # Example: CHILD_CHUNK_SIZE=abc -> we ignore it instead of crashing.
        print(f"[config] '{name}={raw_value}' is not a number. Using {default}.")
        return default


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------

# Path(__file__) is this config.py file; .parent is the project root folder.
# Building paths this way means the project works no matter where you run it from.
PROJECT_ROOT = Path(__file__).parent

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
FAISS_DIR = VECTORSTORE_DIR / "faiss"

# Child chunks and parent chunks are stored here as plain JSON. We need them on
# disk because keyword (BM25) search and parent lookup happen outside the
# vector database.
DOC_STORE_PATH = VECTORSTORE_DIR / "doc_store.json"


# ---------------------------------------------------------------------------
# OpenAI models
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Centralising model names means you can swap "gpt-4o-mini" for another model
# in one place instead of hunting through the codebase.
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# 0.0 makes the model as predictable as possible, which is what we want for
# question answering over documents (we want facts, not creative writing).
LLM_TEMPERATURE = 0.0


# ---------------------------------------------------------------------------
# Vector database
# ---------------------------------------------------------------------------

# "faiss" (local, no account needed) or "pinecone" (remote, needs an API key).
VECTOR_DB = os.getenv("VECTOR_DB", "faiss").strip().lower()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "advanced-rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# Parent chunks are big: they give the LLM enough surrounding context to answer.
PARENT_CHUNK_SIZE = _get_int("PARENT_CHUNK_SIZE", 1600)
PARENT_CHUNK_OVERLAP = _get_int("PARENT_CHUNK_OVERLAP", 200)

# Child chunks are small: short, focused text embeds into a sharper vector,
# so searching over children finds the right spot more accurately.
CHILD_CHUNK_SIZE = _get_int("CHILD_CHUNK_SIZE", 400)
CHILD_CHUNK_OVERLAP = _get_int("CHILD_CHUNK_OVERLAP", 60)


# ---------------------------------------------------------------------------
# Retrieval pipeline
# ---------------------------------------------------------------------------

# How many alternative phrasings of the question the LLM should write.
NUM_QUERY_VARIANTS = _get_int("NUM_QUERY_VARIANTS", 3)

# How many chunks each individual search (vector or keyword) returns.
VECTOR_TOP_K = _get_int("VECTOR_TOP_K", 6)
BM25_TOP_K = _get_int("BM25_TOP_K", 6)

# How many fused candidates survive to the reranking step.
HYBRID_TOP_K = _get_int("HYBRID_TOP_K", 10)

# How many documents the cross-encoder keeps after reordering them.
RERANK_TOP_K = _get_int("RERANK_TOP_K", 4)

# Cross-encoder scores are roughly "how well does this answer the question".
# Anything below this is treated as noise and dropped during refinement.
# Typical values: a good match scores +2 to +10, junk scores -8 or lower.
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-4.0"))

# Rough limit on how much text we send to the LLM. ~4 characters per token,
# so 8000 characters is about 2000 tokens of context.
MAX_CONTEXT_CHARS = _get_int("MAX_CONTEXT_CHARS", 8000)

# Which reranker to use:
#   "cross-encoder" - accurate, needs sentence-transformers + PyTorch.
#   "simple"        - keyword overlap only, needs no extra packages. Use this
#                     on machines where PyTorch will not install or load.
# With "cross-encoder" the app still falls back to the simple reranker
# automatically if the model cannot be loaded, so it never crashes.
RERANKER = os.getenv("RERANKER", "cross-encoder").strip().lower()

# A small, fast reranking model that runs locally on the CPU.
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# How many previous question/answer pairs are fed back into the prompt.
MEMORY_WINDOW = _get_int("MEMORY_WINDOW", 4)


def check_openai_key() -> None:
    """
    Make sure an OpenAI API key is available before we try to call OpenAI.

    Without this check the first API call fails with a long, confusing stack
    trace. Failing early with a readable message is much friendlier.

    Raises:
        ValueError: If OPENAI_API_KEY is missing or empty.
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set.\n"
            "Copy .env.example to .env and paste your OpenAI API key into it."
        )


def check_pinecone_config() -> None:
    """
    Make sure Pinecone settings exist before we try to use Pinecone.

    This only matters when VECTOR_DB=pinecone. With the default FAISS setup the
    project never calls this function, so you do not need a Pinecone account.

    Raises:
        ValueError: If the Pinecone API key or index name is missing.
    """
    if not PINECONE_API_KEY:
        raise ValueError(
            "VECTOR_DB=pinecone but PINECONE_API_KEY is empty.\n"
            "Add your Pinecone key to .env, or set VECTOR_DB=faiss to stay local."
        )
    if not PINECONE_INDEX_NAME:
        raise ValueError("VECTOR_DB=pinecone but PINECONE_INDEX_NAME is empty.")


def describe_settings() -> dict:
    """
    Collect the most interesting settings so the UI can display them.

    The Streamlit sidebar shows this so a learner can see which models and
    which vector database the running app is actually using.

    Returns:
        A dictionary of human-readable setting names and values.
    """
    return {
        "Chat model": OPENAI_CHAT_MODEL,
        "Embedding model": OPENAI_EMBEDDING_MODEL,
        "Vector DB": VECTOR_DB,
        "Parent chunk size": PARENT_CHUNK_SIZE,
        "Child chunk size": CHILD_CHUNK_SIZE,
    }
