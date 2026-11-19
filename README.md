# Local Ollama experiment runner

Hướng dẫn triển khai bằng tiếng Việt: [DEPLOY.md](DEPLOY.md).

This is a reduced copy of the trajectory slicing experiment for a GPU machine.
The `src/` module layout, gold dataset paths, experiment configurations, and
prediction output format match `trajectory-slicing-code`. One runner handles
Qwen, DeepSeek, GLM, or another installed Ollama model.

## Docker job on the GPU machine

Install Docker and Git LFS on the Linux GPU machine. From this directory, after
cloning and downloading the LFS dataset, run these commands on the GPU host
itself so Docker bind mounts its local `outputs/` directory:

```bash
git lfs pull
docker build -t trajectory-slicing-llm-local .
bash run-docker.sh --model qwen3:32b --url http://localhost:11434 --dry-run
```

The dry run reads all inputs and existing checkpoints, prints how many runs are
pending for each configuration, and does not contact Ollama or write results.
The runner refuses to start if `index.json` or a result file listed in it is
missing. This catches a wrong or empty output mount before any inference. Use
`--allow-fresh` only when intentionally starting a model with no prior results.

### Ollama on the host

If Ollama already runs on the GPU host, pull the desired model there and run:

```bash
ollama pull qwen3:32b
bash run-docker.sh --model qwen3:32b --url http://localhost:11434
```

Start `ollama serve` as a host service first if it is not already running. This
mode uses Docker host networking; the Python container itself does not need a
GPU device because the host Ollama process performs inference.

### Ollama in Docker with NVIDIA GPU

If you want Ollama itself in Docker, install the NVIDIA Container Toolkit and
configure the Docker runtime for it. Then run:

```bash
bash run-with-gpu.sh --model qwen3:32b
```

This script checks GPU pass-through with `docker run --gpus all ubuntu nvidia-smi`,
starts the official `ollama/ollama` image with `--gpus all`, pulls the requested
model into a persistent Docker volume, then loads one token at the requested
context length and checks Ollama's reported VRAM usage. A CPU-only load fails
before the experiment starts; partial GPU loading is reported. The script stops
Ollama when the job exits. It checks the output mount before pulling a model. The Python job
shares Ollama's container network. The same script
accepts `deepseek-r1:32b` and `glm-4.7-flash:q8_0` as `--model` values; run them
one at a time. The model volume remains after the containers stop.

Both scripts bind mount this directory's `outputs/` to `/app/outputs`, preserving
the existing checkpoints and writing new results straight to the host. The job
process exits after all configurations are attempted, and `--rm` removes the
job container. If any sample has status `failed`, the job exits with a nonzero
code; rerun the same command to retry it. If Ollama stops serving mid-run, the
job stops without replacing the current sample's checkpoint. Token limit
outcomes remain saved with status `token_limit_exceeded` until you increase
`--context-length`.

To run without Docker, install Python 3.12+ and `uv`, then create `.venv`,
install `requirements.txt`, download NLTK stopwords, and call `src/run_local.py`
with the same `--model` and `--url` arguments.

Use `--context-length N` to override the existing defaults: 32768 for Qwen,
131072 for DeepSeek, and 202752 for GLM. Other model tags default to 32768.
The exact model tag must be present in the Ollama `/api/tags` response.

Results are checkpointed after each sample in
`outputs/experiments/e001_latency_accuracy_tradeoff/<model>/predictions/`.
Rerunning the same command resumes unfinished or failed samples. The runner
executes one sample at a time to avoid competing GPU requests, and a per-model
lock prevents two runner processes from writing to the same checkpoint files.
Existing Qwen, DeepSeek, and GLM prediction checkpoints are included in this
directory and can be committed with ordinary Git. Their JSON files are kept
unchanged so the runner can resume them directly.

The bundled `data/gold/` files are the only input datasets required. No Cloud
Run credentials are used. Transformer based slicing configurations remain
excluded, as in the existing model runners, because they use OpenAI embeddings.

## Git checkout

The 231 MB Trace Elephant JSONL file is tracked through Git LFS; the other
gold JSONL files use ordinary Git. Install Git LFS on the remote machine before
cloning, then run `git lfs install`. After a clone, run `git lfs pull` from the
repository root if the large file was not downloaded automatically. Check that
`data/gold/trace_elephant/trajectories.jsonl` contains JSONL data rather than
an LFS pointer before starting an experiment.
