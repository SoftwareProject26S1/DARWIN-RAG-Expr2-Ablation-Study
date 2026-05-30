"""FAISS index file writer."""

from __future__ import annotations

from pathlib import Path


class FaissIndexWriter:
    """Write normalized vectors to an IndexFlatIP FAISS file."""

    index_backend = "faiss.IndexFlatIP"

    def write(self, path: Path, vectors: list[list[float]]) -> None:
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError(
                "faiss-cpu is required to write FAISS indexes; "
                "install the indexing dependency group first"
            ) from error
        import numpy as np

        if not vectors:
            raise ValueError("cannot write an empty FAISS index")
        array = np.asarray(vectors, dtype="float32")
        index = faiss.IndexFlatIP(int(array.shape[1]))
        index.add(array)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))
