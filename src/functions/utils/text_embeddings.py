# functions/utils/text_embeddings.py
from __future__ import annotations

"""
functions.utils.text_embeddings

Embedding utilities for the HyDE Feed Recommendation POC.

This module is used by **offline/batch** pipelines to:
- embed feed documents for building an index (currently FAISS in POC)
- embed HyDE queries (cached as .npy for online serving)

Online serving MUST NOT call embeddings. Online code should only load cached
embeddings and perform retrieval + deterministic scoring.

Design goals
------------
1) Serving-safe imports
   - Do NOT import google.genai at module import-time.
   - Import it only inside methods that actually call the API.

2) Deterministic numeric behavior
   - Always return np.float32 arrays.
   - Always L2-normalize vectors.
   - Make arrays contiguous (required/optimal for FAISS).
   - Optional dimension truncation ("matryoshka"-style) for index compatibility.

3) Debuggability
   - Optional "uniqueness guard" to detect suspicious embedding collapse:
     many different inputs -> identical/nearly identical vectors.
   - Actionable error messages (include brief input previews).

4) Testability
   - Provide a stable factory `build_embedding_model()` so unit tests can monkeypatch
     a single symbol.

Model choice
------------
Default is "gemini-embedding-001".

Dimension handling
------------------
Some embedding models can return higher-dimensional vectors. This module supports
an optional `output_dim` that truncates vectors BEFORE L2-normalization.

IMPORTANT:
- If you truncate, you must build/search the index using the same truncation.
- Truncation happens BEFORE normalization so cosine similarity remains coherent
  in the truncated space.

Production note
---------------
In production, the "vector DB" will be Vertex AI Vector Search. This module remains
useful for:
- offline experimentation
- building cached query embeddings
- verifying embedding dimensionality/normalization assumptions
"""

import os
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import yaml


# -----------------------------------------------------------------------------
# Credentials
# -----------------------------------------------------------------------------
def _load_google_api_key(credentials_path: str = "parameters/credentials.yaml") -> str:
    """
    Load GOOGLE_API_KEY from a YAML file.

    Precedence in this module:
    - Environment variable GOOGLE_API_KEY (if set) wins
    - Otherwise, read credentials_path YAML

    Rationale
    ---------
    - Cloud Run best practice is env-injected secrets.
    - Local dev can rely on parameters/credentials.yaml.
    """
    env_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if env_key:
        return env_key

    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"{credentials_path} not found and GOOGLE_API_KEY env var not set")

    with open(credentials_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"credentials.yaml must be a dict: {credentials_path}")

    api_key = str(data.get("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY missing/empty (set env var or parameters/credentials.yaml)")

    return api_key


# -----------------------------------------------------------------------------
# Embedding model wrapper
# -----------------------------------------------------------------------------
@dataclass
class GoogleEmbeddingModel:
    """
    Google embedding wrapper (Gemini embeddings).

    Notes
    -----
    - Lazily imports and instantiates the client to avoid import-time failures
      in environments where google-genai isn't installed (tests, minimal serving images).
    - Always returns L2-normalized float32 vectors.
    - Optional `output_dim` truncation for compatibility with an existing index.

    Attributes
    ----------
    model_name:
        Embedding model name (default: "gemini-embedding-001").
    credentials_path:
        YAML credential file path (local dev). Env var GOOGLE_API_KEY overrides.
    output_dim:
        If set, truncate embeddings to this dimension before normalization.
    uniqueness_guard_enabled:
        If True, check that unique inputs produce sufficiently unique vectors.
    uniqueness_guard_min_unique_ratio:
        Threshold for unique-vector / unique-text ratio (default 0.85).
    uniqueness_guard_round_decimals:
        Rounding precision used to create deterministic signatures for uniqueness checks.
    """

    model_name: str = "gemini-embedding-001"
    credentials_path: str = "parameters/credentials.yaml"

    # If set, truncate vectors to this dimension (e.g., 768).
    output_dim: Optional[int] = 768

    # Debug/safety knobs
    uniqueness_guard_enabled: bool = True
    uniqueness_guard_min_unique_ratio: float = 0.85
    uniqueness_guard_round_decimals: int = 8

    # Internal client; typed loosely to avoid import-time dependency
    _client: Optional[object] = None

    # -------------------------
    # Client + response helpers
    # -------------------------
    def _ensure_client(self) -> None:
        """
        Lazily create the google.genai client.

        Keeping the import here prevents tests/serving from failing if google-genai
        is not installed or not needed.
        """
        if self._client is not None:
            return

        from google import genai  # type: ignore

        api_key = _load_google_api_key(self.credentials_path)
        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _as_nonempty_text(text: str) -> str:
        """
        Ensure we never send empty content to the embedding API.
        Some providers reject empty strings; we normalize to a single space.
        """
        t = (text or "").strip()
        return t if t else " "

    @staticmethod
    def _extract_single_vector(resp: Any) -> List[float]:
        """
        Extract one embedding vector from the SDK response.

        Expected:
          resp.embeddings[0].values -> List[float]

        Raises:
          RuntimeError with a clear message if the response shape is unexpected.
        """
        if resp is None:
            raise RuntimeError("Embedding response is None")

        embeddings = getattr(resp, "embeddings", None)
        if not embeddings:
            raise RuntimeError("Embedding response has no embeddings")

        first = embeddings[0]
        values = getattr(first, "values", None)
        if values is None:
            raise RuntimeError("Embedding response embeddings[0] has no 'values'")

        return list(values)

    # -------------------------
    # Vector post-processing
    # -------------------------
    def _maybe_truncate(self, mat: np.ndarray) -> np.ndarray:
        """
        Optionally truncate embedding dimension for index compatibility.

        Truncation happens BEFORE normalization so similarity is meaningful
        in the truncated space.
        """
        if self.output_dim is None:
            return mat

        od = int(self.output_dim)
        if od <= 0:
            raise ValueError(f"output_dim must be positive or None, got {self.output_dim}")

        if mat.ndim != 2:
            raise ValueError("_maybe_truncate expects a 2D array")

        d = int(mat.shape[1])
        if od > d:
            raise ValueError(f"output_dim={od} exceeds embedding dimension d={d} for model={self.model_name}")

        if od == d:
            return mat

        return mat[:, :od]

    @staticmethod
    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        """
        L2-normalize rows of a 2D matrix.

        - Keeps dtype as float32 where possible
        - Avoids division by zero by treating zero-norm rows as norm=1
        """
        if mat.ndim != 2:
            raise ValueError("_l2_normalize expects a 2D array")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return mat / norms

    def _vector_signatures(self, mat: np.ndarray) -> List[bytes]:
        """
        Create stable signatures for each row for uniqueness checks.

        We round to a fixed number of decimals to tolerate tiny floating noise
        while still detecting obvious collapse.
        """
        rounded = np.round(mat, decimals=int(self.uniqueness_guard_round_decimals))
        return [rounded[i].tobytes() for i in range(int(rounded.shape[0]))]

    def _maybe_raise_on_low_uniqueness(self, mat: np.ndarray, texts: List[str], *, task_type: str) -> None:
        """
        Detect suspicious embedding collapse:
          many different inputs -> identical/nearly-identical vectors.

        Comparison is against UNIQUE input texts to avoid false positives when
        the dataset itself contains duplicates.
        """
        if not self.uniqueness_guard_enabled:
            return
        if mat.ndim != 2 or mat.shape[0] <= 1:
            return
        if len(texts) != mat.shape[0]:
            raise RuntimeError(
                f"Uniqueness guard: input/output length mismatch: texts={len(texts)} mat_rows={mat.shape[0]}"
            )

        unique_texts = list(dict.fromkeys(texts))
        n_unique_texts = len(unique_texts)
        if n_unique_texts <= 1:
            return

        sigs = self._vector_signatures(mat)
        n_unique_vecs = len(set(sigs))
        ratio = n_unique_vecs / float(n_unique_texts)

        if ratio < float(self.uniqueness_guard_min_unique_ratio):   # <- 0.85
            preview_n = min(5, n_unique_texts)
            previews = [unique_texts[i][:120].replace("\n", " ") for i in range(preview_n)]
            raise RuntimeError(
                "Embedding uniqueness too low: "
                f"{n_unique_vecs}/{n_unique_texts} ({ratio:.2%}) "
                f"for task_type={task_type} model={self.model_name}. "
                "This MAY indicate embedding collapse or API misuse. "
                f"Examples (first {preview_n} unique inputs): {previews}"
            )

    # -------------------------
    # Public API
    # -------------------------
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed documents for retrieval indexing.

        Parameters
        ----------
        texts:
            List of document strings.

        Returns
        -------
        np.ndarray
            Shape (N, D_out), float32, L2-normalized, contiguous.

        Notes
        -----
        - POC-scale: embeds one-by-one for simplicity and easier debugging.
        - Uses task_type="RETRIEVAL_DOCUMENT".
        """
        if not texts:
            raise ValueError("embed_documents: texts is empty")

        self._ensure_client()

        vectors: List[List[float]] = []
        cleaned: List[str] = []

        for raw in texts:
            text = self._as_nonempty_text(raw)
            cleaned.append(text)

            # SDK call (lazy-import client already initialized)
            resp = self._client.models.embed_content(  # type: ignore[attr-defined]
                model=self.model_name,
                contents=text,
                config={"task_type": "RETRIEVAL_DOCUMENT"},
            )
            vectors.append(self._extract_single_vector(resp))

        mat = np.asarray(vectors, dtype=np.float32)

        # Truncate -> normalize in final space -> contiguous float32 (FAISS friendliness)
        mat = self._maybe_truncate(mat)
        mat = self._l2_normalize(mat)
        mat = np.ascontiguousarray(mat, dtype=np.float32)

        self._maybe_raise_on_low_uniqueness(mat, cleaned, task_type="RETRIEVAL_DOCUMENT")
        return mat

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query for retrieval.

        Parameters
        ----------
        text:
            Query text.

        Returns
        -------
        np.ndarray
            Shape (D_out,), float32, L2-normalized.
        """
        self._ensure_client()

        t = self._as_nonempty_text(text)
        resp = self._client.models.embed_content(  # type: ignore[attr-defined]
            model=self.model_name,
            contents=t,
            config={"task_type": "RETRIEVAL_QUERY"},
        )

        vec = np.asarray(self._extract_single_vector(resp), dtype=np.float32).reshape(1, -1)
        vec = self._maybe_truncate(vec)
        vec = self._l2_normalize(vec)
        return vec[0]


# -----------------------------------------------------------------------------
# Stable public factory (tests monkeypatch this symbol)
# -----------------------------------------------------------------------------
def build_embedding_model(
    *,
    model_name: str = "gemini-embedding-001",
    credentials_path: str = "parameters/credentials.yaml",
    output_dim: Optional[int] = 768,
    uniqueness_guard_enabled: bool = True,
    uniqueness_guard_min_unique_ratio: float = 0.85,
    uniqueness_guard_round_decimals: int = 8,
) -> GoogleEmbeddingModel:
    """
    Stable factory for the embedding model.

    Why this exists
    ---------------
    - Tests can monkeypatch this symbol as the single "embedding entrypoint".
    - Avoids refactor churn when constructor args change.
    """
    return GoogleEmbeddingModel(
        model_name=model_name,
        credentials_path=credentials_path,
        output_dim=output_dim,
        uniqueness_guard_enabled=uniqueness_guard_enabled,
        uniqueness_guard_min_unique_ratio=uniqueness_guard_min_unique_ratio,
        uniqueness_guard_round_decimals=uniqueness_guard_round_decimals,
    )


# -----------------------------------------------------------------------------
# In-memory index (tiny POC helper; not used in serving)
# -----------------------------------------------------------------------------
@dataclass
class EmbeddingIndex:
    """
    Simple in-memory embedding index (POC scale).

    Assumptions
    -----------
    - doc_embeddings are L2-normalized
    - cosine similarity == dot product
    """

    model: GoogleEmbeddingModel
    doc_embeddings: Optional[np.ndarray] = None  # (N, D_out)

    def build(self, documents: List[str]) -> None:
        """Embed and store document vectors in memory."""
        if not documents:
            raise ValueError("EmbeddingIndex.build: documents is empty")
        self.doc_embeddings = self.model.embed_documents(documents)

    def query(self, query_text: str) -> np.ndarray:
        """
        Compute dot-product similarity between stored documents and a query vector.

        Returns:
            np.ndarray shape (N,) of similarity scores.
        """
        if self.doc_embeddings is None:
            raise RuntimeError("EmbeddingIndex.query: index not built/loaded")
        q = self.model.embed_query(query_text)  # (D_out,)
        return self.doc_embeddings @ q


__all__ = [
    "GoogleEmbeddingModel",
    "EmbeddingIndex",
    "build_embedding_model",
]
