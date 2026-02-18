"""Smoke test for Modal baseline setup.

Run with: uv run python test_modal_smoke.py

Tests image build, app hydration, and remote mup imports without running full inference.
"""

import modal

from reliable_monitoring.modal_baseline import app, image


@app.function(image=image, timeout=120)
def smoke_test() -> str:
    """Verify the Modal container can import mup and access config."""
    import os

    checks = []

    # Check env vars
    for var in ("MUP_PROJECT_ROOT", "MUP_CONFIG_DIR", "DATA_DIR"):
        val = os.environ.get(var)
        checks.append(f"{var}={val} (exists={val is not None})")

    # Check mup imports
    from models_under_pressure.config import LOCAL_MODELS

    checks.append(f"LOCAL_MODELS keys: {list(LOCAL_MODELS.keys())}")

    from models_under_pressure.baselines.continuation import likelihood_continuation_prompts

    checks.append(f"Prompt configs: {list(likelihood_continuation_prompts.keys())}")

    from models_under_pressure.experiments.monitoring_cascade import get_abbreviated_model_name

    checks.append(
        "get_abbreviated_model_name('meta-llama/Llama-3.1-8B-Instruct')"
        f"={get_abbreviated_model_name('meta-llama/Llama-3.1-8B-Instruct')}"
    )

    return "\n".join(checks)


if __name__ == "__main__":
    modal.enable_output()
    print("Starting Modal smoke test...")
    print("This tests image build, app hydration, and remote mup imports.\n")

    with app.run():
        result = smoke_test.remote()

    print("=== Remote container output ===")
    print(result)
    print("\nSmoke test passed!")
