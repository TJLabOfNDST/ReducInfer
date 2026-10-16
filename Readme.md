# ReducInfer

ReducInfer stores jointly factorized Transformer weights in PostgreSQL and executes inference with pure SQL. A weight block is represented by a cluster-shared matrix `D` and a block-specific matrix `C`; inference evaluates `(X D) C` directly without reconstructing the original dense weight.

The repository contains the complete production workflow: checkpoint acquisition, bounded-memory CPU or CUDA preprocessing, transactional PostgreSQL import, and SQL Transformer execution.

## Directly supported models

- BERT-Large (`google-bert/bert-large-uncased`)
- RoBERTa-Large (`FacebookAI/roberta-large`)
- ELECTRA-Large (`google/electra-large-discriminator`)
- GPT-2 Medium (`openai-community/gpt2-medium`)
- OPT-350M (`facebook/opt-350m`)
- Pythia-410M (`EleutherAI/pythia-410m`)
- Qwen2.5-0.5B (`Qwen/Qwen2.5-0.5B`)
- SmolLM2-360M (`HuggingFaceTB/SmolLM2-360M`)

## Requirements and installation

Use Python 3.10 or newer and PostgreSQL 15.

Install only the database inference runtime:

```bash
python -m pip install .
```

Install checkpoint acquisition and CPU preprocessing support:

```bash
python -m pip install '.[models,preprocess]'
```

Add the CUDA 12 preprocessing backend:

```bash
python -m pip install '.[models,preprocess,cuda]'
```

The `models` extra contains Hugging Face Hub, safetensors, and PyTorch. PyTorch is used only to read BF16 safetensors and official legacy `pytorch_model.bin` checkpoints. The `preprocess` extra contains SciPy and FAISS; the `cuda` extra contains CuPy. None of these extras is required after factors have been imported.

For a local disposable database:

```bash
docker run -d --name reducinfer-postgres \
  -p 5432:5432 \
  -e POSTGRES_HOST_AUTH_METHOD=trust \
  postgres:15

export REDUCINFER_DSN='postgresql://postgres@127.0.0.1:5432/postgres'
```

Set `HF_TOKEN` only when a Hugging Face repository requires authentication. ReducInfer never writes either environment variable into an artifact or log.

## Complete workflow

Initialize the schema:

```bash
reducinfer init-db
```

To discard and recreate only the `reducinfer` schema, explicitly confirm the database name:

```bash
reducinfer init-db --allow-reset-database postgres
```

Download all eight checkpoints:

```bash
reducinfer download --model all --checkpoint-dir checkpoints
```

Create CPU factors with the default 1024 block size and rank 256:

```bash
reducinfer reduce \
  --model all \
  --checkpoint-dir checkpoints \
  --artifact-dir artifacts \
  --block-size 1024 \
  --rank 256 \
  --backend cpu
```

For CUDA preprocessing, replace the backend and select the device and bounded block batch:

```bash
reducinfer reduce \
  --model all \
  --checkpoint-dir checkpoints \
  --artifact-dir artifacts \
  --block-size 1024 \
  --rank 256 \
  --backend cuda \
  --device 0 \
  --batch-blocks 256
```

Import the matching artifact and remaining one-dimensional parameters in one transaction:

```bash
reducinfer import-db \
  --model all \
  --checkpoint-dir checkpoints \
  --artifact-dir artifacts \
  --block-size 1024 \
  --rank 256 \
  --backend cpu
```

Use `--backend cuda` when importing artifacts produced by the CUDA command. Import validates the ordered model set, checkpoint revision and size, block size, rank, topology, and referenced files before changing database rows. Replacing an imported model does not reset unrelated models.

## Inference

Inputs and outputs are NumPy `.npy` files loaded with pickling disabled.

- `hidden`: float32 shape `(batch, sequence, hidden_width)`; runs all stored layers.
- `tokens`: integer shape `(batch, sequence)`; values are normalized to int64 before embeddings and all stored layers run.
- `prefill`: float32 hidden states with batch size 1 and a non-empty request ID; replaces that request's KV cache.
- `decode`: float32 shape `(1, 1, hidden_width)`, a request ID, and a non-negative position; appends one incremental step.

```bash
reducinfer infer \
  --model google-bert/bert-large-uncased \
  --mode hidden \
  --input hidden.npy \
  --output output.npy

reducinfer infer \
  --model Qwen/Qwen2.5-0.5B \
  --mode prefill \
  --request-id request-1 \
  --input prompt_hidden.npy \
  --output prompt_output.npy

reducinfer infer \
  --model Qwen/Qwen2.5-0.5B \
  --mode decode \
  --request-id request-1 \
  --position 8 \
  --input next_hidden.npy \
  --output next_output.npy
```

## Artifacts and execution boundary

CPU preprocessing writes a versioned manifest, BRP features, cluster labels, and cluster factors. CUDA preprocessing additionally writes a contiguous C buffer, block topology, and tensor manifest. Raw checkpoint tensors are not copied into the artifact directory.

Python validates inputs, scans checkpoints, imports persistent values, submits SQL, and returns the final array. From input materialization through Transformer operators to the final database result, model computation remains pure SQL: the system does not use UDFs, stored procedures, or external PostgreSQL extensions.

Generated checkpoints, artifacts, `.npy` files, logs, and local environments are ignored by the repository and can be removed after deployment when they are no longer needed.
