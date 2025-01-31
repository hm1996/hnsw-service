import os
import hnswlib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import psycopg2
from psycopg2.extras import Json

# FastAPI app
app = FastAPI()

# Configuration
INDEX_PATH = "hnsw_index.bin"
dim = 128  # Dimension of vector
max_elements = 10000  # Maximum elements in the index

# Initialize or load HNSWLIB Index
index = hnswlib.Index(space="l2", dim=dim)
if os.path.exists(INDEX_PATH):
    index.load_index(INDEX_PATH)
else:
    index.init_index(max_elements=max_elements, ef_construction=200, M=32)
index.set_ef(100)

# PostgreSQL connection
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="admin",
    host="localhost",
    port=5432,
)
cursor = conn.cursor()

# Define database schema (run this once to set up the database)
cursor.execute("""
CREATE TABLE IF NOT EXISTS vectors (
    id SERIAL PRIMARY KEY,
    vector REAL[],
    metadata JSONB
);
""")
conn.commit()

# Pydantic models
class AddVectorRequest(BaseModel):
    vector: List[float]
    metadata: Optional[dict] = None

class KNNQueryRequest(BaseModel):
    query_vector: List[float]
    k: int = 5

# Endpoint to add a vector to the database and HNSW index
@app.post("/add-vector")
def add_vector(request: AddVectorRequest):
    vector = np.array(request.vector, dtype=np.float32)

    if len(vector) != dim:
        raise HTTPException(
            status_code=400,
            detail=f"Vector dimension mismatch. Expected {dim}, got {len(vector)}.",
        )

    # Insert into PostgreSQL
    cursor.execute(
        "INSERT INTO vectors (vector, metadata) VALUES (%s, %s) RETURNING id;",
        (vector.tolist(), Json(request.metadata)),
    )
    vector_id = cursor.fetchone()[0]
    conn.commit()

    # Add to HNSW index
    index.add_items([vector], [vector_id])

    # Save index to disk
    index.save_index(INDEX_PATH)

    return {"message": "Vector added successfully", "id": vector_id}

# Endpoint to perform a KNN search
@app.post("/knn-search")
def knn_search(request: KNNQueryRequest):
    query_vector = np.array(request.query_vector, dtype=np.float32)

    if len(query_vector) != dim:
        raise HTTPException(
            status_code=400,
            detail=f"Query vector dimension mismatch. Expected {dim}, got {len(query_vector)}.",
        )

    # Perform KNN search using HNSWLIB
    labels, distances = index.knn_query(query_vector, k=request.k)

    # Fetch metadata from PostgreSQL
    cursor.execute(
        "SELECT id, metadata FROM vectors WHERE id = ANY(%s);", (labels[0].tolist(),)
    )
    results = cursor.fetchall()

    return {
        "results": [
            {"id": res[0], "metadata": res[1], "distance": dist}
            for res, dist in zip(results, distances[0].tolist())
        ]
    }

# Endpoint to rebuild the HNSW index from the database
@app.post("/rebuild-index")
def rebuild_index():
    cursor.execute("SELECT id, vector FROM vectors;")
    rows = cursor.fetchall()

    index.init_index(max_elements=max_elements, ef_construction=200, M=16)
    for row in rows:
        index.add_items([row[1]], [row[0]])

    # Save index to disk
    index.save_index(INDEX_PATH)

    return {"message": "Index rebuilt successfully", "total_vectors": len(rows)}

@app.post("/knn")
async def knn_search(request: KNNQueryRequest):
    query = np.array(request.query_vector, dtype=np.float32)
    
    # Check if the query vector dimensions match the index dimensions
    if len(query) != dim:
        raise HTTPException(
            status_code=400, 
            detail=f"Query vector dimension mismatch. Expected {dim}, got {len(query)}."
        )
    
    # Perform KNN search
    labels, distances = index.knn_query(query, k=request.k)
    return {"labels": labels.tolist(), "distances": distances.tolist()}
