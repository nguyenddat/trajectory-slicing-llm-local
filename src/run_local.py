#!/usr/bin/env python
"""Run the existing attribution/slicing experiment against a local Ollama host."""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run as base_run
from core.chat_models import configure_ollama
from data.final_schema import Trajectory
from method_failure_attribution.all_at_once import AllAtOnceMethod
from method_failure_attribution.step_by_step import (
    FutureOnlyFixedStepSlicing,
    PastOnlyFixedStepSlicing,
    StepByStepMethod,
)
from method_slicing.tokenization.bow import BOW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTEXT_LENGTHS = {
    "qwen3:32b": 32768,
    "deepseek-r1:32b": 131072,
    "glm-4.7-flash:q8_0": 202752,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exact Ollama model tag")
    parser.add_argument("--url", required=True, help="Ollama base URL, e.g. http://localhost:11434")
    parser.add_argument("--context-length", type=int, help="Override Ollama num_ctx")
    parser.add_argument("--dry-run", action="store_true", help="Show resume counts without Ollama or file writes")
    parser.add_argument("--allow-fresh", action="store_true", help="Permit a new output directory without checkpoints")
    parser.add_argument("--check-gpu", action="store_true", help="Load one token and verify Ollama uses GPU memory")
    args = parser.parse_args()
    if args.context_length is not None and args.context_length <= 0:
        parser.error("--context-length must be positive")
    return args


def run_configs(model_name: str):
    # Transformer slicing calls OpenRouter embeddings. Keep its definitions in
    # run.py for compatibility, but do not schedule those settings locally.
    slicing_configs = [
        (name, cls, {"model_name": model_name} if name == "llm_based" else params)
        for name, cls, params in base_run.SLICING_CONFIGS
        if not name.startswith(("c99_transformer_", "text_tiling_transformer_"))
    ]
    step_configs = [
        (name, cls, {"model_name": model_name} if name == "llm_based" else params)
        for name, cls, params in base_run.STEP_BY_STEP_SLICING_CONFIGS
        if not name.startswith(("c99_transformer_", "text_tiling_transformer_"))
    ] + [
        (f"fixed_step_{direction}_{steps}", cls, {"max_steps_per_slice": steps})
        for direction, cls in (
            ("past", PastOnlyFixedStepSlicing),
            ("future", FutureOnlyFixedStepSlicing),
        )
        for steps in (5, 7, 9, 11, 13)
    ]
    return [
        (name, cls, params, "all_at_once", AllAtOnceMethod)
        for name, cls, params in slicing_configs
    ] + [
        (name, cls, params, "step_by_step", StepByStepMethod)
        for name, cls, params in step_configs
    ]


def wait_for_model(model_name: str, url: str) -> None:
    deadline = time.monotonic() + 30 * 60
    with httpx.Client(base_url=url, timeout=30) as client:
        while True:
            try:
                response = client.get("/api/tags")
                response.raise_for_status()
                available = {item["name"] for item in response.json()["models"]}
                if model_name in available:
                    return
                raise ValueError(
                    f"Ollama model {model_name!r} is not installed at {url}. "
                    f"Available: {sorted(available)}"
                )
            except httpx.TransportError as error:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Ollama at {url} was unavailable for 30 minutes") from error
                print(f"Waiting for Ollama at {url}: {error}", flush=True)
                time.sleep(15)


def load_samples() -> list[tuple[str, int, Trajectory, str]]:
    samples = []
    for input_path in base_run.INPUT_PATHS:
        source_name = input_path.relative_to(PROJECT_ROOT).as_posix()
        with input_path.open(encoding="utf-8") as source:
            for index, line in enumerate(line for line in source if line.strip()):
                trajectory = Trajectory.model_validate_json(line)
                if (
                    trajectory.mistake_step is not None
                    and str(trajectory.mistake_step).strip().lower() not in {"", "none"}
                    and trajectory.mistake_agent
                ):
                    samples.append((source_name, index, trajectory, f"{source_name}:{index}"))
    return samples


def read_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Empty checkpoint file: {path}")
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in checkpoint: {path}")
    return value


def should_run(sample: dict | None, context_length: int) -> bool:
    if not isinstance(sample, dict):
        return True
    if sample.get("status") == "token_limit_exceeded":
        return sample.get("context_length", 0) < context_length
    return not (sample.get("status") == "complete" and isinstance(sample.get("prediction"), dict))


def is_token_limit_error(error: dict) -> bool:
    message = str(error.get("message", "")).lower()
    return base_run._is_token_limit_error(error) or any(
        marker in message
        for marker in (
            "token count", "too many tokens", "input token limit", "request too large",
            "exceeds the available context size", "prompt is longer than the context length",
        )
    )


def is_service_error(error: Exception) -> bool:
    if isinstance(error, httpx.TransportError):
        return True
    if getattr(error, "status_code", None) in {404, 500, 502, 503, 504}:
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "connection refused", "failed to connect", "service unavailable",
            "status code: 503", "model not found", "model is not loaded",
        )
    )


def check_gpu(model_name: str, url: str, context_length: int) -> None:
    wait_for_model(model_name, url)
    with httpx.Client(base_url=url, timeout=600) as client:
        response = client.post(
            "/api/generate",
            json={
                "model": model_name,
                "prompt": "OK",
                "stream": False,
                "options": {"num_ctx": context_length, "num_predict": 1},
            },
        )
        response.raise_for_status()
        response = client.get("/api/ps")
        response.raise_for_status()
        loaded = next(
            (item for item in response.json()["models"] if item.get("name") == model_name),
            None,
        )
    if loaded is None:
        raise RuntimeError(f"Ollama did not keep {model_name} loaded after the GPU check")
    vram = loaded.get("size_vram", 0)
    if vram <= 0:
        raise RuntimeError(f"Ollama loaded {model_name} on CPU only")
    print(f"Ollama GPU check: {model_name}, {vram:,} of {loaded['size']:,} bytes in GPU memory")


def main() -> None:
    args = parse_args()
    model_name = args.model
    context_length = args.context_length or DEFAULT_CONTEXT_LENGTHS.get(model_name, 32768)
    model_dir = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-.")
    if not model_dir:
        raise ValueError("Model tag cannot be converted to an output directory")
    if args.check_gpu:
        check_gpu(model_name, args.url, context_length)
        return
    results_dir = (
        PROJECT_ROOT / "outputs" / "experiments" / "e001_latency_accuracy_tradeoff"
        / model_dir / "predictions"
    )
    index_path = results_dir / "index.json"

    if not index_path.is_file() and not args.allow_fresh:
        raise FileNotFoundError(
            f"Missing checkpoint {index_path}; mount the existing outputs directory "
            "or pass --allow-fresh to start a new experiment"
        )
    samples = load_samples()
    manifest = read_json(index_path, {"version": 1, "configs": {}})
    if not isinstance(manifest.get("configs"), dict):
        raise ValueError(f"Invalid config manifest: {index_path}")
    if any(entry.get("model") != model_name for entry in manifest["configs"].values()):
        raise ValueError(f"Output directory {results_dir} contains results for another model")
    for entry in manifest["configs"].values():
        if not (results_dir / entry["file"]).is_file():
            raise FileNotFoundError(
                f"Missing result file {results_dir / entry['file']}; checkpoint is incomplete"
            )
    if not args.dry_run:
        results_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform != "win32":
            import fcntl

            lock_file = (results_dir / ".runner.lock").open("a+")
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Another runner is writing to {results_dir}") from error
        wait_for_model(model_name, args.url)
        configure_ollama(args.url, context_length)
    failed_samples = 0
    total_pending = 0

    for slicing_name, slicing_class, params, attribution_name, method_class in run_configs(model_name):
        config_id = f"{slicing_name}__{attribution_name}"
        config_entry = manifest["configs"].setdefault(
            config_id,
            {
                "file": f"{config_id}.json",
                "slicing": slicing_name,
                "slicing_parameters": params,
                "attribution": attribution_name,
                "model": model_name,
            },
        )
        if attribution_name == "step_by_step":
            config_entry["context_direction"] = base_run._step_by_step_context_type(slicing_class)
        result_path = results_dir / config_entry["file"]
        result = read_json(result_path, {"config": config_entry, "samples": {}})
        if not isinstance(result.get("config"), dict) or not isinstance(result.get("samples"), dict):
            raise ValueError(f"Invalid result checkpoint: {result_path}")
        if result["config"].get("model") != model_name:
            raise ValueError(f"Result file {result_path} belongs to another model")

        slice_range_cache = {}
        if attribution_name == "step_by_step" and slicing_name in base_run.REUSABLE_SLICE_CONFIGS:
            source_entry = manifest["configs"].get(f"{slicing_name}__all_at_once")
            if source_entry and source_entry.get("slicing_parameters") == params:
                slice_range_cache = base_run._load_slice_range_cache(results_dir / source_entry["file"])

        pending = []
        for source_name, index, trajectory, sample_key in samples:
            if sample_key not in result["samples"] and sample_key.replace("/", "\\") in result["samples"]:
                sample_key = sample_key.replace("/", "\\")
            if should_run(result["samples"].get(sample_key), context_length):
                pending.append((source_name, index, trajectory, sample_key))
        total_pending += len(pending)
        if args.dry_run:
            print(f"{config_id}: {len(samples) - len(pending)} already done, {len(pending)} pending")
            continue
        if not pending:
            failed_samples += sum(sample.get("status") == "failed" for sample in result["samples"].values())
            print(f"{config_id}: {len(samples)}/{len(samples)} done, 0 pending", flush=True)
            continue

        config_entry["context_length"] = context_length
        base_run._write_json_atomic(index_path, manifest)
        result["config"]["context_length"] = context_length

        slicing_params = params.copy()
        if slicing_params.get("tokenizer") == "bow":
            slicing_params["tokenizer"] = BOW()
        method = method_class(
            model_name=model_name,
            slicing_method=slicing_class(**slicing_params) if slicing_class else None,
        )
        already_done = len(samples) - len(pending)
        print(f"{config_id}: {already_done}/{len(samples)} done, {len(pending)} pending", flush=True)
        for position, (source_name, index, trajectory, sample_key) in enumerate(pending, start=already_done + 1):
            print(f"{config_id}: starting {position}/{len(samples)} {sample_key}", flush=True)
            started_at = time.monotonic()
            sample = {"source": source_name, "index": index, "context_length": context_length}
            try:
                cached_ranges = base_run._cached_slice_ranges_for_sample(slice_range_cache, sample_key)
                prediction = base_run._predict_with_slice_cache(method, trajectory, attribution_name, cached_ranges)
                sample["prediction"] = prediction.model_dump()
                sample["metrics"] = base_run.eval(trajectory, prediction, index)
                sample["status"] = "complete"
            except Exception as error:  # Isolate and save each sample failure.
                if is_service_error(error):
                    raise RuntimeError(f"Ollama stopped serving {model_name}; checkpoint preserved") from error
                error_info = {"type": type(error).__name__, "message": str(error)}
                sample["status"] = "token_limit_exceeded" if is_token_limit_error(error_info) else "failed"
                sample["prediction"] = None
                sample["metrics"] = {"step_accuracy": 0, "agent_accuracy": 0} if sample["status"] == "token_limit_exceeded" else None
                sample["error"] = error_info
            try:
                sample["slice_ranges"] = base_run._slice_ranges(method, trajectory)
            except Exception as error:
                sample["slice_ranges"] = []
                sample["slice_ranges_error"] = {"type": type(error).__name__, "message": str(error)}
            result["samples"][sample_key] = sample
            base_run._write_json_atomic(result_path, result)
            base_run._write_json_atomic(index_path, manifest)
            elapsed = time.monotonic() - started_at
            print(
                f"{config_id}: {position}/{len(samples)} {sample['status']} "
                f"{sample_key} ({elapsed:.1f}s)",
                flush=True,
            )
        failed_samples += sum(sample.get("status") == "failed" for sample in result["samples"].values())
        print(f"Saved {len(pending)} samples to {result_path}")

    if args.dry_run:
        print(f"Total: {total_pending} pending model/config/sample runs")
        return
    if failed_samples:
        raise SystemExit(f"Finished with {failed_samples} failed samples; rerun to retry them")


if __name__ == "__main__":
    main()
