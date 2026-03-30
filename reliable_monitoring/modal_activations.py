"""Modal-based activation computation + reduction for cloud GPU execution.

Shares the same Modal app and infrastructure as :mod:`modal_baseline`.
"""

import numpy as np

from reliable_monitoring.modal_baseline import (
    _hf_secret,
    _timeout,
    _volumes,
    app,
    image,
    model_cache,
)

ACTIVATION_GPU_MAP = {
    "llama-1b": "A10G",
    "llama-3b": "A10G",
    "llama-8b": "A10G",
    "llama-70b": "A100-80GB:2",
    "gemma-1b": "A10G",
    "gemma-12b": "A10G",
    "gemma-27b": "A100-80GB:2",
}


def _compute_activations_impl(
    model_name: str,
    dataset,
    layer: int,
    reduction_strategy: str,
    batch_size: int = 8,
    chunk_size: int = 500,
) -> list[list[float]]:
    """Compute reduced activations on a remote GPU.

    Processes the dataset in chunks to avoid OOM: for each chunk, computes
    raw activations via forward pass, reduces immediately (collapsing the
    sequence dimension), and frees the raw activations before proceeding
    to the next chunk.
    """
    import torch
    from models_under_pressure.model import LLMModel

    from reliable_monitoring.reductions import get_reduction_function

    model = LLMModel.load(model_name, batch_size=batch_size)
    reduction_fn = get_reduction_function(reduction_strategy)

    n = len(dataset)
    reduced_chunks: list[torch.Tensor] = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = dataset[start:end]
        print(f"Processing chunk {start}–{end} of {n}")

        # Forward pass → (n_layers, chunk_size, seq_len, hidden_dim)
        activations, inputs = model.get_batched_activations_for_layers(
            dataset=chunk,
            layers=[layer],
        )

        acts = activations[0]  # (chunk_size, seq_len, hidden_dim)
        attention_mask = inputs["attention_mask"]
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask, device=acts.device)
        elif attention_mask.device != acts.device:
            attention_mask = attention_mask.to(acts.device)

        # Apply attention mask (zero out padding positions)
        acts = acts * attention_mask.unsqueeze(-1)

        # Reduce immediately → (chunk_size, hidden_dim)
        reduced = reduction_fn(acts, attention_mask).cpu()
        reduced_chunks.append(reduced)

        # Free GPU memory
        del activations, inputs, acts, attention_mask

    model_cache.commit()

    result = torch.cat(reduced_chunks, dim=0)
    return result.numpy().tolist()


# One Modal function per GPU tier — GPU is fixed at definition time in Modal.


@app.function(image=image, gpu="T4", volumes=_volumes, secrets=[_hf_secret], timeout=_timeout)
def _compute_activations_t4(model_name, dataset, layer, reduction_strategy, batch_size=16):
    return _compute_activations_impl(model_name, dataset, layer, reduction_strategy, batch_size)


@app.function(image=image, gpu="A10G", volumes=_volumes, secrets=[_hf_secret], timeout=_timeout)
def _compute_activations_a10g(model_name, dataset, layer, reduction_strategy, batch_size=16):
    return _compute_activations_impl(model_name, dataset, layer, reduction_strategy, batch_size)


@app.function(image=image, gpu="A100-80GB", volumes=_volumes, secrets=[_hf_secret], timeout=_timeout)
def _compute_activations_a100(model_name, dataset, layer, reduction_strategy, batch_size=16):
    return _compute_activations_impl(model_name, dataset, layer, reduction_strategy, batch_size)


@app.function(image=image, gpu="A100-80GB:2", volumes=_volumes, secrets=[_hf_secret], timeout=_timeout)
def _compute_activations_a100x2(model_name, dataset, layer, reduction_strategy, batch_size=16):
    return _compute_activations_impl(model_name, dataset, layer, reduction_strategy, batch_size)


_GPU_DISPATCH = {
    "T4": _compute_activations_t4,
    "A10G": _compute_activations_a10g,
    "A100-80GB": _compute_activations_a100,
    "A100-80GB:2": _compute_activations_a100x2,
}


def compute_activations_modal(
    model_name: str,
    dataset,  # LabelledDataset
    layer: int,
    reduction_strategy: str,
    *,
    batch_size: int = 8,
    gpu: str | None = None,
) -> np.ndarray:
    """Compute reduced activations on a Modal cloud GPU.

    Args:
        model_name: Full HuggingFace model name (e.g. "meta-llama/Llama-3.2-1B-Instruct").
        dataset: LabelledDataset instance (without activation fields).
        layer: Layer number to extract activations from.
        reduction_strategy: Reduction strategy name (e.g. "mean").
        batch_size: Batch size for model forward pass.
        gpu: Override GPU type. If None, auto-selects based on model.

    Returns:
        Reduced activations as numpy array of shape ``(n_samples, hidden_dim)``.
    """
    from models_under_pressure.experiments.monitoring_cascade import get_abbreviated_model_name

    from reliable_monitoring.modal_baseline import _ensure_app_running

    # Strip heavy tensor fields to reduce network payload
    cols_to_drop = [c for c in ("activations", "attention_mask", "input_ids") if c in dataset.other_fields]
    if cols_to_drop:
        dataset = dataset.drop_cols(*cols_to_drop)

    # Auto-select GPU based on model size (activation extraction needs more VRAM)
    if gpu is None:
        abbreviated = get_abbreviated_model_name(model_name)
        gpu = ACTIVATION_GPU_MAP.get(abbreviated, "A10G")

    fn = _GPU_DISPATCH.get(gpu, _compute_activations_a10g)

    _ensure_app_running()
    result = fn.remote(
        model_name=model_name,
        dataset=dataset,
        layer=layer,
        reduction_strategy=reduction_strategy,
        batch_size=batch_size,
    )
    return np.array(result)
