import numpy as np
from sqlmodel import Session, select
from models.local import ReferenceChunks

def match_reference_chunks( session: Session, query_embedding: list[float],match_count: int = 5,
    filter_dict: dict = None) -> list[dict]:
    statement = select(ReferenceChunks)
    all_chunks = session.exec(statement).all()
    if filter_dict:
        candidates = [chunk for chunk in all_chunks if all((chunk.metadata_json or {}).get(k) == v for k, v in filter_dict.items())]
    else:
        candidates = all_chunks
    
    if not candidates:
       return []
    
    query_vec = np.array(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm > 0:
        query_vec = query_vec / query_norm

    matrix = np.array([np.frombuffer(c.embedding, dtype=np.float32) for c in candidates])
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix_norms[matrix_norms == 0] = 1.0
    matrix = matrix / matrix_norms

    
    similarities = np.dot(matrix, query_vec)
    top_indices = np.argsort(similarities)[::-1][:match_count]
    return [
    {"chunk": candidates[i], "similarity": float(similarities[i])}
    for i in top_indices
]