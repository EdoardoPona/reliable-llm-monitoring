import json

import numpy as np
import pytest
from mixture_shift_report import domain_metrics, read_run, summarise

DOMAINS = ["anthropic", "medical", "toolace"]


def record(seed=42, selected=True, risk="budget"):
    return {
        "expert": "strong",
        "alpha": 0.3,
        "method": "mixture",
        "model_seed": seed,
        "split_seed": seed,
        "guaranteed_risk": risk,
        "selection": {"threshold": 0.1 if selected else None},
        "evaluation": {
            "domains": {
                g: {"accuracy_error": error, "budget": budget}
                for g, error, budget in zip(DOMAINS, [0.1, 0.2, 0.3], [0.1, 0.2, 0.4], strict=True)
            },
            "accuracy_error": {"worst_domain": 0.3},
            "budget": {"worst_domain": 0.4},
        }
        if selected
        else None,
    }


@pytest.mark.parametrize(
    "risk,margin,complementary",
    [("budget", [0.2, 0.1, -0.1], [0.9, 0.8, 0.7]), ("accuracy_error", [0.2, 0.1, 0.0], [0.1, 0.2, 0.4])],
)
def test_margin_sign_and_complementary_metric(risk, margin, complementary):
    got = domain_metrics(record(risk=risk), DOMAINS)
    assert got is not None
    np.testing.assert_allclose(got[0], margin)
    np.testing.assert_allclose(got[1], complementary)
    assert domain_metrics(record(selected=False), DOMAINS) is None


def test_summary_preserves_nonselection_and_weights_named_mixtures():
    mixtures = {"domains": DOMAINS, "reference": [0.5, 0.25, 0.25]}
    result = summarise([record(), record(seed=43, selected=False)], mixtures)[0]
    assert (result["n_certified"], result["n_trials"]) == (1, 2)
    assert result["unselected"] == [{"model_seed": 43, "split_seed": 43}]
    assert result["named_mixtures"]["ToolACE"][0]["expert_call_rate"] == pytest.approx(0.4)
    assert result["named_mixtures"]["Original benchmark"][0]["accuracy"] == pytest.approx(0.825)
    assert result["per_selected_policy"][0]["worst_domain_accuracy"] == pytest.approx(0.7)
    empty = summarise([record(selected=False)], mixtures)[0]
    assert empty["n_certified"] == 0
    assert empty["per_selected_policy"] == []
    assert all(values == [] for values in empty["named_mixtures"].values())


def test_report_rejects_incomplete_runs(tmp_path):
    (tmp_path / "inventory.json").write_text(json.dumps({"status": "incomplete", "inventory_only": False}))
    with pytest.raises(ValueError, match="completed calibration"):
        read_run(tmp_path)
