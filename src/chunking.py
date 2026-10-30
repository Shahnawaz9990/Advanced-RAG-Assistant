from typing import Dict, List, Tuple
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except OSError as e:
    raise ImportError(f"Error while import in chunking module: {e}")
import config
from src.ingestion import guess_section


def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Build a text splitter that tries to cut text at meaningful boundaries.

    "Meaning-aware" splitting here means the splitter prefers to break at a
    paragraph break, then at a line break, then at the end of a sentence, and
    only chops mid-word as a last resort. That keeps ideas intact inside a
    chunk instead of slicing a sentence in half.

    Args:
        chunk_size: Target chunk length in characters.
        chunk_overlap: How many characters neighbouring chunks share, so a
            sentence sitting on a boundary is not lost.

    Returns:
        A configured RecursiveCharacterTextSplitter.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
        length_function = len,
    )    


def split_documents(documents: List[Document]) -> Tuple[List[Document], List[Document]]:
    """
    Turn loaded documents into large parent chunks and small child chunks.

    This is the core of parent-child retrieval. Each child carries a
    `parent_id` in its metadata, which is the link we follow later to recover
    the bigger context.

    Args:
        documents: Cleaned documents from `src.ingestion`.

    Returns:
        A tuple `(parents, children)`:
            parents  - large chunks, used as the context given to the LLM.
            children - small chunks, the ones we actually embed and search.

    Example:
        A 5-page handbook may produce 12 parents and 48 children. Searching
        finds child #37; the LLM reads parent "handbook.pdf::8" instead.
    """
    parent_splitter = make_splitter(config.PARENT_CHUNK_SIZE, config.PARENT_CHUNK_OVERLAP)
    child_spitter = make_splitter(config.CHILD_CHUNK_SIZE, config.CHILD_CHUNK_OVERLAP)

    parents: List[Document] = []
    children: List[Document] = []

    # We count parents separately for each file. Numbering per file (rather
    # than across the whole batch) means a chunk always gets the same id no
    # matter which other documents were processed alongside it - which is what
    # lets `update_index` recognise and skip a file it has already indexed.
    parents_per_file: Dict[str, int] = {}

    for document in documents:
        filename = document.metadata.get("filename", "unknown")

        for parent_text in parent_splitter.split_text(document.page_content):
            parent_number = parents_per_file.get(filename, 0)
            parents_per_file[filename] = parent_number + 1
            parent_id = f"{filename}::{parent_number}"

            parent_metadata = dict(document.metadata)
            parent_metadata["parent_id"] = parent_id
            parent_metadata["section"] = guess_section(parent_text)

            parents.append(Document(page_content = parent_text, metadata = parent_metadata))

            for child_index, child_text in enumerate(child_spitter.split_text(parent_text)):
                child_metadata = dict(parent_metadata)
                child_metadata["chunk_id"] = f"{parent_id}::{child_index}"
                children.append(Document(page_content=child_text, metadata = child_metadata))

    return parents, children


def build_parent_lookup(parents: List[Document]) -> Dict[str, Document]:
    """
    Build a dictionary that maps a parent_id to its parent Document.

    Retrieval gives us child chunks; we need an instant way to fetch the
    matching parent. A dictionary lookup is O(1), which is as simple and fast
    as this problem needs to be.

    Args:
        parents: The list of parent chunks.

    Returns:
        A dictionary like {"handbook.pdf::3": Document(...), ...}.
    """
    return {parent.metadata["parent_id"]: parent for parent in parents}    