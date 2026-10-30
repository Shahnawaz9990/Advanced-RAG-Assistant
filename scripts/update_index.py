"""
Add new documents to an index that already exists.

Run it with:
    python scripts/update_index.py

Use this after dropping a new PDF into `data/documents`. Unlike
`build_index.py`, this keeps everything that is already indexed and only
embeds the chunks it has not seen before, which is faster and cheaper.

Chunks are recognised by an id built from the filename, so running this twice
in a row adds nothing the second time.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from src.indexing import update_index  # noqa: E402
from src.ingestion import load_documents_from_folder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    """
    Load the documents folder and add anything new to the existing index.

    Prints how many chunks were added and how many were skipped because they
    were already indexed.
    """
    print(f"Reading documents from: {config.DOCUMENTS_DIR}")

    documents = load_documents_from_folder(config.DOCUMENTS_DIR)
    if not documents:
        print("No readable documents found. Add PDF, TXT or DOCX files and try again.")
        return

    try:
        stats = update_index(documents)
    except Exception as error:
        print(f"Update failed: {error}")
        return

    if "added_children" in stats:
        print(f"\nAdded {stats['added_children']} new child chunks.")
        print(f"Skipped {stats['skipped_children']} chunks that were already indexed.")
    else:
        # There was no index yet, so update_index() built one from scratch.
        print(f"\nNo index existed, so a new one was built with {stats['children']} chunks.")


if __name__ == "__main__":
    main()
