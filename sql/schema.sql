CREATE SCHEMA IF NOT EXISTS reducinfer;
SET search_path TO reducinfer, public;

CREATE TABLE IF NOT EXISTS models (
    model_key SMALLSERIAL PRIMARY KEY,
    model_id TEXT NOT NULL UNIQUE,
    architecture TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tensors (
    tensor_id SERIAL PRIMARY KEY,
    model_key SMALLINT NOT NULL REFERENCES models(model_key) ON DELETE CASCADE,
    tensor_name TEXT NOT NULL,
    rows INTEGER NOT NULL CHECK (rows > 0),
    cols INTEGER NOT NULL CHECK (cols > 0),
    UNIQUE (model_key, tensor_name)
);

CREATE TABLE IF NOT EXISTS shared_matrix_d_columns (
    d_id INTEGER NOT NULL,
    rank_idx SMALLINT NOT NULL CHECK (rank_idx >= 0),
    d_values REAL[] NOT NULL,
    PRIMARY KEY (d_id, rank_idx),
    CHECK (cardinality(d_values) > 0 AND cardinality(d_values) <= 2048)
);

CREATE TABLE IF NOT EXISTS block_slice (
    block_key SERIAL PRIMARY KEY,
    tensor_id INTEGER NOT NULL REFERENCES tensors(tensor_id) ON DELETE CASCADE,
    factor_id INTEGER NOT NULL,
    block_row_offset INTEGER NOT NULL CHECK (block_row_offset >= 0),
    block_col_offset INTEGER NOT NULL CHECK (block_col_offset >= 0),
    c_start INTEGER NOT NULL CHECK (c_start >= 0),
    valid_rows SMALLINT NOT NULL CHECK (valid_rows > 0),
    valid_cols SMALLINT NOT NULL CHECK (valid_cols > 0)
);

CREATE INDEX IF NOT EXISTS block_slice_tensor_position_idx
    ON block_slice (tensor_id, block_row_offset, block_col_offset);

CREATE TABLE IF NOT EXISTS layer_c_vectors (
    tensor_id INTEGER NOT NULL REFERENCES tensors(tensor_id) ON DELETE CASCADE,
    factor_id INTEGER NOT NULL,
    block_row_offset INTEGER NOT NULL CHECK (block_row_offset >= 0),
    output_col INTEGER NOT NULL CHECK (output_col >= 0),
    rank SMALLINT NOT NULL CHECK (rank > 0),
    c_values REAL[] NOT NULL,
    PRIMARY KEY (tensor_id, block_row_offset, output_col),
    CHECK (cardinality(c_values) = rank)
);

CREATE INDEX IF NOT EXISTS layer_c_vectors_factor_lookup_idx
    ON layer_c_vectors (tensor_id, factor_id, block_row_offset);

CREATE TABLE IF NOT EXISTS dense_vectors (
    model_key SMALLINT NOT NULL REFERENCES models(model_key) ON DELETE CASCADE,
    vector_name TEXT NOT NULL,
    values REAL[] NOT NULL,
    PRIMARY KEY (model_key, vector_name)
);

CREATE TABLE IF NOT EXISTS model_layers (
    model_key SMALLINT NOT NULL REFERENCES models(model_key) ON DELETE CASCADE,
    layer_id INTEGER NOT NULL CHECK (layer_id >= 0),
    template TEXT NOT NULL,
    query_heads SMALLINT NOT NULL CHECK (query_heads > 0),
    kv_heads SMALLINT NOT NULL CHECK (kv_heads > 0),
    head_dim SMALLINT NOT NULL CHECK (head_dim > 0),
    position_encoding TEXT NOT NULL,
    norm_kind TEXT NOT NULL,
    norm_epsilon DOUBLE PRECISION NOT NULL CHECK (norm_epsilon > 0),
    activation TEXT NOT NULL,
    tensor_prefix TEXT NOT NULL CHECK (length(tensor_prefix) > 0),
    rope_theta DOUBLE PRECISION,
    attention_bias BOOLEAN NOT NULL DEFAULT FALSE,
    mlp_bias BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (model_key, layer_id),
    CHECK (query_heads % kv_heads = 0),
    CHECK ((position_encoding = 'rope') = (rope_theta IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS kv_cache (
    request_id TEXT NOT NULL,
    layer_id INTEGER NOT NULL,
    head_id SMALLINT NOT NULL,
    position_idx INTEGER NOT NULL CHECK (position_idx >= 0),
    key_values REAL[] NOT NULL,
    value_values REAL[] NOT NULL,
    PRIMARY KEY (request_id, layer_id, head_id, position_idx)
);
