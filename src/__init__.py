"""
The `src` package holds one module per pipeline step:

    ingestion.py   - load and clean documents
    chunking.py    - split into parent and child chunks
    indexing.py    - embed and store the chunks
    retrieval.py   - multi-query + hybrid search + parent recovery
    reranking.py   - cross-encoder reranking
    refinement.py  - pick the final context and build citations
    memory.py      - conversation history
    rag_pipeline.py- glue the steps together
"""
