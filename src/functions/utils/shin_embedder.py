import os
import numpy as np
from google import genai

def embed_texts_gemini(texts:list[str], output_dim:int=768, task_type: str = "RETRIEVAL_DOCUMENT", embedding_model:str = "gemini-embedding-001")->np.ndarray:
    if not texts:
        return np.zeros((0,output_dim or 0), dtype=np.float32)
    # step01 : API key
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY is not set")
    # step02 : client creation
    client = genai.Client(api_key=key)
    vectors = []
    for raw in texts:
        text = raw.strip() or " "
        resp = client.models.embed_content(
            model=embedding_model,
            contents=text,
            config={"task_type": task_type},
        )
        if not hasattr(resp, "embeddings") or not resp.embeddings:
            raise RuntimeError("Invalid embedding response")
        values = resp.embeddings[0].values
        vectors.append(values)
    # step03 : ndarray + float32 (friend logic)
    mat = np.asarray(vectors, dtype=np.float32)
    # step04 : optional truncation (before normalize)
    if output_dim is not None:
        if output_dim > mat.shape[1]:
            raise ValueError(
                f"output_dim {output_dim} > embedding dim {mat.shape[1]}"
            )
        mat = mat[:, :output_dim]
    # step05 : L2 normalize (friend logic)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    mat = mat / norms

    return mat

# texts = [
#     "Hello, world!",
#     "Vertex AI is great for embeddings.",
#     "I hate everything about you",
#     "21 gungs"
# ]

# vectors = embed_texts_gemini(
#     texts,
#     output_dim=768,
# )

# print(vectors.shape)      # (2, 768)
# print(vectors.dtype)      # float32
# print(np.linalg.norm(vectors[0]))  # ~1.0


# import asyncio


# async def _embed_one(client, text: str, task_type: str):

#     text = text.strip() or " "

#     resp = client.models.embed_content(
#         model="gemini-embedding-001",
#         contents=text,
#         config={"task_type": task_type},
#     )

#     if not hasattr(resp, "embeddings") or not resp.embeddings:
#         raise RuntimeError("Invalid embedding response")

#     return resp.embeddings[0].values


# async def embed_texts_gemini_async(
#     texts: list[str],
#     output_dim: int = 768,
#     task_type: str = "RETRIEVAL_DOCUMENT",
#     max_concurrency: int = 5,
# ) -> np.ndarray:

#     if not texts:
#         return np.zeros((0, output_dim or 0), dtype=np.float32)

#     # step01 : API key
#     key = os.getenv("GOOGLE_API_KEY")
#     if not key:
#         raise ValueError("GOOGLE_API_KEY is not set")

#     # step02 : async client
#     client = genai.Client(api_key=key)

#     sem = asyncio.Semaphore(max_concurrency)

#     async def worker(text):
#         async with sem:
#             return await _embed_one(client, text, task_type)

#     tasks = [worker(t) for t in texts]

#     vectors = await asyncio.gather(*tasks)

#     # step03 : ndarray
#     mat = np.asarray(vectors, dtype=np.float32)

#     # step04 : truncation
#     if output_dim is not None:
#         if output_dim > mat.shape[1]:
#             raise ValueError(
#                 f"output_dim {output_dim} > embedding dim {mat.shape[1]}"
#             )
#         mat = mat[:, :output_dim]

#     # step05 : L2 normalize
#     norms = np.linalg.norm(mat, axis=1, keepdims=True)
#     norms = np.where(norms == 0.0, 1.0, norms)
#     mat = mat / norms

#     return mat