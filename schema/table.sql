CREATE TABLE vectors (
    id SERIAL PRIMARY KEY,      -- Auto-incrementing ID
    vector REAL[],              -- Array for storing the vector embedding
    metadata JSONB              -- Metadata associated with the vector (JSON format)
);
