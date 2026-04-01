"""Modal-based baseline LLM inference for cloud GPU execution."""

import os

import modal
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.environ["DATA_DIR"]

app = modal.App("reliable-llm-monitoring-baseline")
model_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

MUP_REPO = "https://github.com/EdoardoPona/models-under-pressure.git"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.50.0",
        "accelerate>=1.4.0",
        "numpy<2",
        "pydantic>=2.0",
        "pydantic-settings>=2.8.1",
        "python-dotenv>=1.0.1",
        "pyyaml>=6.0.2",
        "tqdm>=4.67.1",
        "jaxtyping>=0.2.38",
        "scikit-learn>=1.6.1",
        "pandas>=2.2.3",
        "datasets>=3.3.2",
        "huggingface>=0.0.1",
        "h5py>=3.13.0",
        "einops>=0.8.1",
        "hydra-core>=1.3.0",
        "openai>=1.0.0",
        "zstandard>=0.23.0",
        "boto3>=1.35.0",
        "wandb>=0.19.0",
        "bitsandbytes>=0.45.0",
    )
    .run_commands(
        f"git clone {MUP_REPO} /root/mup",
        "cd /root/mup && pip install --no-deps .",
    )
    .env(
        {
            "MUP_PROJECT_ROOT": "/root/mup",
            "MUP_CONFIG_DIR": "/root/mup/config",
            "DATA_DIR": "/root/mup/data",
            "HF_HOME": "/root/.cache/huggingface",
        }
    )
    # Local mounts must be last — Modal disallows build steps after them
    .add_local_python_source("reliable_monitoring")
    .add_local_dir(f"{DATA_DIR}/inputs", remote_path="/root/mup/data/inputs")
)

_hf_secret = modal.Secret.from_name("huggingface-secret")
_volumes = {"/root/.cache/huggingface": model_cache}
_timeout = 3600

GPU_MAP = {
    "llama-1b": "T4",
    "llama-3b": "T4",
    "llama-8b": "A10G",
    "llama-70b": "A100-80GB:2",
    "gemma-1b": "T4",
    "gemma-12b": "A10G",
    "gemma-27b": "A100-80GB:2",
}

# Models large enough to require 4-bit quantization to fit in VRAM
_QUANTIZE_4BIT = {"llama-70b", "gemma-27b"}


_cached_baseline_model: dict = {}


def _run_baseline_impl(
    baseline_model_name: str,
    dataset,
    baseline_batch_size: int = 16,
) -> list[float]:
    """Shared inference implementation called by all GPU-specific Modal functions."""
    from models_under_pressure.baselines.continuation import (
        LikelihoodContinuationBaseline,
        likelihood_continuation_prompts,
    )
    from models_under_pressure.config import LOCAL_MODELS
    from models_under_pressure.experiments.monitoring_cascade import (
        get_abbreviated_model_name,
        get_model_baseline_prompt,
    )
    from models_under_pressure.model import LLMModel

    abbreviated = get_abbreviated_model_name(baseline_model_name)
    prompt_key = get_model_baseline_prompt(abbreviated)
    prompt_config = likelihood_continuation_prompts[prompt_key]

    # Reuse model across calls in the same container
    if _cached_baseline_model.get("name") == abbreviated:
        model = _cached_baseline_model["model"]
    else:
        model_kwargs = {}
        if abbreviated in _QUANTIZE_4BIT:
            import torch
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                model_kwargs["max_memory"] = {i: "40GiB" for i in range(n_gpus)}

        model = LLMModel.load(LOCAL_MODELS[abbreviated], model_kwargs=model_kwargs)
        _cached_baseline_model["name"] = abbreviated
        _cached_baseline_model["model"] = model

    baseline_model = LikelihoodContinuationBaseline(model, prompt_config=prompt_config)

    results = baseline_model.likelihood_classify_dataset(
        dataset,
        batch_size=baseline_batch_size,
    )
    model_cache.commit()
    return list(results.other_fields["high_stakes_score"])


# One Modal function per GPU tier — GPU is fixed at definition time in Modal.


@app.function(image=image, gpu="T4", volumes=_volumes, secrets=[_hf_secret], timeout=_timeout)
def _run_baseline_t4(baseline_model_name, dataset, baseline_batch_size=16):
    return _run_baseline_impl(baseline_model_name, dataset, baseline_batch_size)


@app.function(image=image, gpu="A10G", volumes=_volumes, secrets=[_hf_secret], timeout=_timeout)
def _run_baseline_a10g(baseline_model_name, dataset, baseline_batch_size=16):
    return _run_baseline_impl(baseline_model_name, dataset, baseline_batch_size)


@app.function(image=image, gpu="A100-80GB", volumes=_volumes, secrets=[_hf_secret], timeout=_timeout)
def _run_baseline_a100(baseline_model_name, dataset, baseline_batch_size=16):
    return _run_baseline_impl(baseline_model_name, dataset, baseline_batch_size)


@app.function(image=image, gpu="A100-80GB:2", volumes=_volumes, secrets=[_hf_secret], timeout=_timeout)
def _run_baseline_a100x2(baseline_model_name, dataset, baseline_batch_size=16):
    return _run_baseline_impl(baseline_model_name, dataset, baseline_batch_size)


_GPU_DISPATCH = {
    "T4": _run_baseline_t4,
    "A10G": _run_baseline_a10g,
    "A100-80GB": _run_baseline_a100,
    "A100-80GB:2": _run_baseline_a100x2,
}


_app_context = None


def _ensure_app_running():
    """Lazily start the Modal app and keep it alive for the process lifetime."""
    global _app_context
    if _app_context is None:
        import atexit

        _app_context = app.run()
        _app_context.__enter__()
        atexit.register(_shutdown_app)


def _shutdown_app():
    """Clean up the Modal app context on process exit."""
    global _app_context
    if _app_context is not None:
        _app_context.__exit__(None, None, None)
        _app_context = None


def run_llm_baseline_modal(
    baseline_model_name: str,
    dataset,  # LabelledDataset
    baseline_batch_size: int = 16,
    gpu: str | None = None,
) -> np.ndarray:
    """Run LLM baseline on a Modal cloud GPU.

    Strips heavy tensor fields from the dataset before sending to reduce
    network payload, selects an appropriate GPU based on model size, and
    calls the correct GPU-specific Modal function.

    Args:
        baseline_model_name: Full HuggingFace model name.
        dataset: LabelledDataset instance.
        baseline_batch_size: Batch size for inference.
        gpu: Override GPU type (e.g. "A100-80GB"). If None, auto-selects based on model.

    Returns:
        Numpy array of high-stakes probability scores.
    """
    from models_under_pressure.experiments.monitoring_cascade import get_abbreviated_model_name

    # Strip heavy tensor fields to reduce network payload
    cols_to_drop = [c for c in ("activations", "attention_mask", "input_ids") if c in dataset.other_fields]
    if cols_to_drop:
        dataset = dataset.drop_cols(*cols_to_drop)

    # Auto-select GPU based on model size
    if gpu is None:
        abbreviated = get_abbreviated_model_name(baseline_model_name)
        gpu = GPU_MAP.get(abbreviated, "A10G")

    fn = _GPU_DISPATCH.get(gpu, _run_baseline_a10g)

    _ensure_app_running()
    scores = fn.remote(
        baseline_model_name=baseline_model_name,
        dataset=dataset,
        baseline_batch_size=baseline_batch_size,
    )
    return np.array(scores)
