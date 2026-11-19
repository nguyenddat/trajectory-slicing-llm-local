#!/usr/bin/env python

import json
import re
import sys
from pathlib import Path

# Keep the existing absolute imports working for both
# `python src/run.py` and `python -m src.run`.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tqdm import tqdm

from data.final_schema import Trajectory
from method_failure_attribution.all_at_once import AllAtOnceMethod
from method_failure_attribution.base import FailureAttributionOutput
from method_failure_attribution.step_by_step import (
    FutureOnlyFixedStepSlicing,
    PastOnlyFixedStepSlicing,
    StepByStepMethod,
)
from method_slicing.c99 import C99Slice
from method_slicing.fixed_step import FixedStepSlicing
from method_slicing.fixed_token import FixedTokenSlicing
from method_slicing.llm_based import LlmBasedSlicing
from method_slicing.text_tiling import TextTilingSlice
from method_slicing.tokenization.bow import BOW
from method_slicing.tokenization.transformer import Transformer
from utils.google_sheets import create_sink as _create_google_sheets_sink

MISSING_LABELS_PATH = (
    Path(__file__).resolve().parents[1] / "results" / "missing_failure_labels.jsonl"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "openai/gpt-4o-mini-2024-07-18"
INPUT_PATHS = (
    PROJECT_ROOT / "data/gold/agentrx/magentic_one.jsonl",
    PROJECT_ROOT / "data/gold/agentrx/tau_retail.jsonl",
    PROJECT_ROOT / "data/gold/trace_elephant/trajectories.jsonl",
    PROJECT_ROOT / "data/gold/who_when/algorithm_generated.jsonl",
    PROJECT_ROOT / "data/gold/who_when/hand_crafted.jsonl",
)
RESULTS_DIR = Path(__file__).resolve().parent / "results"
INDEX_PATH = RESULTS_DIR / "index.json"

REUSABLE_SLICE_CONFIGS = frozenset(
    {
        "c99_bow_rank5_min3",
        "text_tiling_bow_window2_min3",
    }
)


def eval(trajectory: Trajectory, prediction: FailureAttributionOutput, index: int):
    missing_labels = []
    if trajectory.mistake_step is None or str(trajectory.mistake_step).strip().lower() in {"", "none"}:
        missing_labels.append("mistake_step")

    if not trajectory.mistake_agent:
        missing_labels.append("mistake_agent")

    if missing_labels:
        MISSING_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MISSING_LABELS_PATH.open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(
                    {
                        "index": index,
                        "dataset": trajectory.dataset,
                        "reason": f"Cannot evaluate: missing {', '.join(missing_labels)}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        return {"step_accuracy": 1, "agent_accuracy": 1}

    step_match = re.fullmatch(r"(?:step\s+)?(\d+)", str(trajectory.mistake_step).strip(), re.IGNORECASE)
    if step_match is None:
        raise ValueError(f"Invalid ground-truth step: {trajectory.mistake_step!r}")

    ground_truth_step = int(step_match.group(1))
    if prediction.pred_step_index == -1:
        return {"step_accuracy": 0, "agent_accuracy": 0}
    if not 1 <= prediction.pred_step_index <= len(trajectory.behaviors):
        raise ValueError(f"Invalid predicted index: {prediction.pred_step_index}")

    behavior = trajectory.behaviors[prediction.pred_step_index - 1]
    pred_agent_name = behavior.agent_name or behavior.role

    def normalize_agent(name: str | None) -> str:
        if name is None:
            return ""
        name = re.sub(r"\s*\([^)]*\)\s*$", "", name.strip())
        return " ".join(name.casefold().split())

    normalized_pred_agent = normalize_agent(pred_agent_name)
    return {
        "step_accuracy": int(behavior.step_index == ground_truth_step),
        "agent_accuracy": int(
            bool(normalized_pred_agent)
            and normalized_pred_agent == normalize_agent(trajectory.mistake_agent)
        ),
    }


SLICING_CONFIGS = [
    ("no_slicing", None, {}),
] + [
    (f"fixed_step_{steps}", FixedStepSlicing, {"max_steps_per_slice": steps})
    for steps in (5, 7, 9, 11, 13)
] + [
    (
        f"fixed_step_{steps}_overlap{steps // 2}",
        FixedStepSlicing,
        {"max_steps_per_slice": steps, "overlapping": steps // 2},
    )
    for steps in (5, 7, 9, 11, 13)
] + [
    (f"fixed_token_{tokens}", FixedTokenSlicing, {"max_tokens_per_slice": tokens})
    for tokens in (8000, 15000)
] + [
    (
        f"fixed_token_{tokens}_overlap{tokens // 2}",
        FixedTokenSlicing,
        {"max_tokens_per_slice": tokens, "overlapping": tokens // 2},
    )
    for tokens in (8000, 15000)
] + [
    ("llm_based", LlmBasedSlicing, {"model_name": MODEL_NAME})
] + [
    (
        f"c99_{tokenizer}_rank{rank_window}_min{min_slice_steps}",
        C99Slice,
        {
            "tokenizer": tokenizer,
            "rank_window": rank_window,
            "min_slice_steps": min_slice_steps,
        },
    )
    for tokenizer in ("bow", "transformer")
    for rank_window in (5,)
    for min_slice_steps in (3,)
] + [
    (
        f"c99_{tokenizer}_rank{rank_window}_min{min_slice_steps}_extra_context{extra_context}",
        C99Slice,
        {
            "tokenizer": tokenizer,
            "rank_window": rank_window,
            "min_slice_steps": min_slice_steps,
            "extra_context": extra_context,
        },
    )
    for tokenizer in ("bow", "transformer")
    for rank_window in (5,)
    for min_slice_steps in (3,)
    for extra_context in (1,)
] + [
    (
        f"text_tiling_{tokenizer}_window{window_step}_min{min_slice_steps}",
        TextTilingSlice,
        {
            "tokenizer": tokenizer,
            "window_step": window_step,
            "smoothing_width": 3,
            "smoothing_rounds": 1,
            "min_slice_steps": min_slice_steps,
        },
    )
    for tokenizer in ("bow", "transformer")
    for window_step in (2,)
    for min_slice_steps in (3,)
] + [
    (
        f"text_tiling_{tokenizer}_window{window_step}_min{min_slice_steps}_extra_context{extra_context}",
        TextTilingSlice,
        {
            "tokenizer": tokenizer,
            "window_step": window_step,
            "smoothing_width": 3,
            "smoothing_rounds": 1,
            "min_slice_steps": min_slice_steps,
            "extra_context": extra_context,
        },
    )
    for tokenizer in ("bow", "transformer")
    for window_step in (2,)
    for min_slice_steps in (3,)
    for extra_context in (1,)
]

ATTRIBUTION_METHODS = {
    "all_at_once": AllAtOnceMethod,
    "step_by_step": StepByStepMethod,
}

STEP_BY_STEP_SLICING_CONFIGS = [
    ("no_slicing", None, {}),
] + [
    (
        f"fixed_step_{direction}_{steps}",
        slicing_class,
        {"max_steps_per_slice": steps},
    )
    for direction, slicing_class in (
        ("centered", FixedStepSlicing),
        ("past", PastOnlyFixedStepSlicing),
        ("future", FutureOnlyFixedStepSlicing),
    )
    for steps in (5, 7, 9, 11, 13)
] + [
    ("llm_based", LlmBasedSlicing, {"model_name": MODEL_NAME})
] + [
    (
        f"c99_{tokenizer}_rank{rank_window}_min{min_slice_steps}",
        C99Slice,
        {
            "tokenizer": tokenizer,
            "rank_window": rank_window,
            "min_slice_steps": min_slice_steps,
        },
    )
    for tokenizer in ("bow", "transformer")
    for rank_window in (5,)
    for min_slice_steps in (3,)
] + [
    (
        f"text_tiling_{tokenizer}_window{window_step}_min{min_slice_steps}",
        TextTilingSlice,
        {
            "tokenizer": tokenizer,
            "window_step": window_step,
            "smoothing_width": 3,
            "smoothing_rounds": 1,
            "min_slice_steps": min_slice_steps,
        },
    )
    for tokenizer in ("bow", "transformer")
    for window_step in (2,)
    for min_slice_steps in (3,)
]


def main() -> None:
    sheet_sink = _create_google_sheets_sink()
    all_combinations = [
        # Transformer-based settings remain defined but are skipped because
        # their embedding model uses OpenAI credits.
        (slicing_name, slicing_class, params, "step_by_step")
        for slicing_name, slicing_class, params in STEP_BY_STEP_SLICING_CONFIGS
        if not slicing_name.startswith(("c99_transformer_", "text_tiling_transformer_"))
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    index_text = INDEX_PATH.read_text(encoding="utf-8").strip() if INDEX_PATH.exists() else ""
    manifest = json.loads(index_text) if index_text else {}
    manifest.setdefault("version", 1)
    manifest.setdefault("configs", {})
    combinations = all_combinations
    samples = []
    for input_path in INPUT_PATHS:
        with input_path.open(encoding="utf-8") as source:
            trajectories = [
                Trajectory.model_validate_json(line) for line in source if line.strip()
            ]
        source_name = input_path.relative_to(PROJECT_ROOT).as_posix()
        dataset_group = input_path.parent.name
        labeled_trajectories = [
            (index, trajectory)
            for index, trajectory in enumerate(trajectories)
            if (
                trajectory.mistake_step is not None
                and str(trajectory.mistake_step).strip().lower() not in {"", "none"}
                and bool(trajectory.mistake_agent)
            )
        ]
        skipped_count = len(trajectories) - len(labeled_trajectories)
        if skipped_count:
            print(f"Skipping {skipped_count} unlabeled samples in {source_name}")
        samples.extend(
            (source_name, dataset_group, index, trajectory)
            for index, trajectory in labeled_trajectories
        )

    for slicing_name, slicing_class, params, attribution_name in combinations:
            config_id = f"{slicing_name}__{attribution_name}"
            config_entry = manifest["configs"].setdefault(
                config_id,
                {
                    "file": f"{config_id}.json",
                    "slicing": slicing_name,
                    "slicing_parameters": params,
                    "attribution": attribution_name,
                    "model": MODEL_NAME,
                },
            )
            if attribution_name == "step_by_step":
                config_entry["context_direction"] = _step_by_step_context_type(
                    slicing_class
                )
            _write_json_atomic(INDEX_PATH, manifest)
            result_path = RESULTS_DIR / config_entry["file"]
            if result_path.exists():
                result_text = result_path.read_text(encoding="utf-8").strip()
                if result_text:
                    try:
                        result = json.loads(result_text)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid JSON in result file {result_path}: {error}"
                        ) from error
                else:
                    result = {"config": config_entry, "samples": {}}
            else:
                result = {"config": config_entry, "samples": {}}
            result.setdefault("config", config_entry)
            result.setdefault("samples", {})
            samples_compacted = _compact_samples(result["samples"])
            for saved_sample_key, saved_sample in result["samples"].items():
                _append_google_sheet_row(
                    sheet_sink, MODEL_NAME, config_id, saved_sample_key, saved_sample
                )

            pending = []
            for source_name, dataset_group, index, trajectory in samples:
                sample_key = f"{source_name}:{index}"
                if sample_key not in result["samples"]:
                    old_key = sample_key.replace("/", "\\")
                    if old_key in result["samples"]:
                        sample_key = old_key
                if _sample_should_run(result["samples"].get(sample_key)):
                    pending.append((source_name, dataset_group, index, trajectory, sample_key))
            completed_count = len(samples) - len(pending)
            progress = tqdm(
                pending,
                desc=config_id,
                unit="sample",
                total=len(samples),
                initial=completed_count,
            )
            if not pending:
                if samples_compacted:
                    _write_json_atomic(result_path, result)
                progress.close()
                continue

            slicing_params = params.copy()
            if "tokenizer" in slicing_params:
                slicing_params["tokenizer"] = (
                    BOW() if params["tokenizer"] == "bow" else Transformer()
                )
            method_kwargs = {
                "model_name": MODEL_NAME,
                "slicing_method": (
                    slicing_class(**slicing_params)
                    if slicing_class is not None
                    else None
                ),
            }
            method = ATTRIBUTION_METHODS[attribution_name](**method_kwargs)
            for source_name, dataset_group, index, trajectory, sample_key in progress:
                sample = {
                    "source": source_name,
                    "index": index,
                }
                try:
                    prediction = method.predict(trajectory)
                    sample["prediction"] = prediction.model_dump()
                    sample["metrics"] = eval(trajectory, prediction, index)
                    sample["status"] = "complete"
                except Exception as error:  # noqa: BLE001 - isolate one failed sample
                    error_info = {
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                    token_limit_exceeded = _is_token_limit_error(error_info)
                    sample["status"] = (
                        "token_limit_exceeded"
                        if token_limit_exceeded
                        else "failed"
                    )
                    sample["prediction"] = None
                    sample["metrics"] = (
                        {"step_accuracy": 0, "agent_accuracy": 0}
                        if token_limit_exceeded
                        else None
                    )
                    sample["error"] = error_info
                try:
                    sample["slice_ranges"] = _slice_ranges(method, trajectory)
                except Exception as error:  # noqa: BLE001 - persist the sample either way
                    sample["slice_ranges"] = []
                    sample["slice_ranges_error"] = {
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                    if sample["status"] == "complete":
                        sample["status"] = "failed"
                result["samples"][sample_key] = sample
                _write_json_atomic(result_path, result)
                _write_json_atomic(INDEX_PATH, manifest)
                _append_google_sheet_row(sheet_sink, MODEL_NAME, config_id, sample_key, sample)
            print(f"Saved {len(pending)} samples to {result_path}")


def _append_google_sheet_row(sink, model: str, config_id: str, sample_key: str, sample: dict) -> None:
    if sink is None:
        return
    try:
        sink.append(model=model, config_id=config_id, sample_key=sample_key, sample=sample)
    except Exception as error:  # noqa: BLE001 - sheet outages must not lose local results
        import warnings

        warnings.warn(f"Could not sync {sample_key} to Google Sheets: {error}", stacklevel=2)


def _step_by_step_context_type(slicing_class) -> str:
    if slicing_class is None:
        return "past"
    if issubclass(slicing_class, FixedStepSlicing):
        return getattr(slicing_class, "context_direction", "centered")
    return "semantic"


def _sample_should_run(sample: dict | None) -> bool:
    """Run new samples and retry only failures unrelated to model token limits."""
    if sample is None:
        return True
    if isinstance(sample, dict) and isinstance(sample.get("prediction"), dict):
        return False
    return bool(
        isinstance(sample, dict)
        and sample.get("status") == "failed"
        and not _is_token_limit_error(sample.get("error"))
    )


def _is_token_limit_error(error: dict | None) -> bool:
    if not isinstance(error, dict):
        return False
    message = str(error.get("message", "")).lower()
    return (
        error.get("type") == "OpenAIContextOverflowError"
        or "context_length_exceeded" in message
        or "maximum context length" in message
        or "token limit exceeded" in message
    )


def _slice_ranges(method, trajectory: Trajectory) -> list[list[int]]:
    """Convert actual slices to zero-based, half-open ranges."""
    slices = getattr(method, "last_slices", None)
    if slices is None:
        return []
    positions = {
        id(behavior): position
        for position, behavior in enumerate(trajectory.behaviors)
    }
    ranges = []
    for current_slice in slices:
        if not current_slice:
            continue
        indices = [positions[id(behavior)] for behavior in current_slice]
        ranges.append([min(indices), max(indices) + 1])
    return ranges


def _load_slice_range_cache(path: Path) -> dict[str, list[list[int]]]:
    """Load persisted slice ranges for deterministic, compatible slicers."""
    if not path.exists():
        return {}
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    samples = result.get("samples", {})
    if not isinstance(samples, dict):
        return {}
    return {
        sample_key: sample["slice_ranges"]
        for sample_key, sample in samples.items()
        if isinstance(sample, dict)
        and isinstance(sample.get("slice_ranges"), list)
        and not sample.get("slice_ranges_error")
    }


def _cached_slice_ranges_for_sample(
    cache: dict[str, list[list[int]]], sample_key: str
) -> list[list[int]] | None:
    ranges = cache.get(sample_key)
    if ranges is None:
        ranges = cache.get(sample_key.replace("/", "\\"))
    return ranges


def _predict_with_slice_cache(method, trajectory, attribution_name, slice_ranges):
    if attribution_name == "step_by_step":
        return method.predict(
            trajectory,
            precomputed_slice_ranges=slice_ranges,
        )
    return method.predict(trajectory)


def _compact_samples(samples: dict[str, dict]) -> bool:
    """Drop duplicated trajectory metadata from results created by older runs."""
    changed = False
    for sample in samples.values():
        if isinstance(sample, dict):
            for key in ("dataset", "dataset_group", "trajectory"):
                if key in sample:
                    sample.pop(key)
                    changed = True
    return changed


def _write_json_atomic(path: Path, value: dict) -> None:
    """Persist one JSON checkpoint without leaving a half-written result file."""
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


if __name__ == "__main__":
    main()
