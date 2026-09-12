"""Tests for the routing balance intervention knob.

``scripts/run_routing_balance_intervention.py`` needs a deployed YOLO-Master
checkout to train, so these tests cover its pure knob surface with plain stubs.
The contract under test: the intervention sets ``moe_loss_fn.balance_loss_coeff``
and leaves the module-level ``balance_loss_coeff`` alone, because only the loss
function attribute actually changes the training objective.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_routing_balance_intervention import (
    DEFAULT_INTERVENTION_COEFF,
    assert_balance_coeff,
    read_balance_coeffs,
    tune_balance_coeff,
)


class _LossFn:
    def __init__(self, coeff: float = 1.0) -> None:
        self.balance_loss_coeff = coeff


class _Block:
    """Stand-in for a routed MoE block that owns a MoELoss."""

    def __init__(self, name: str, coeff: float = 1.0, with_loss_fn: bool = True) -> None:
        self._name = name
        self.balance_loss_coeff = coeff
        if with_loss_fn:
            self.moe_loss_fn = _LossFn(coeff)


class _Model:
    def __init__(self, children: list) -> None:
        self._children = children

    def named_modules(self):
        yield "", self
        for block in self._children:
            yield block._name, block


def test_default_intervention_coefficient_is_four() -> None:
    assert DEFAULT_INTERVENTION_COEFF == 4.0


def test_tune_sets_only_the_moe_loss_fn_attribute() -> None:
    blocks = [_Block("model.5"), _Block("model.8"), _Block("model.11")]
    rows = tune_balance_coeff(_Model(blocks), 4.0)

    assert len(rows) == 3
    for block in blocks:
        assert block.moe_loss_fn.balance_loss_coeff == 4.0
        # the module-level attribute must NOT be touched
        assert block.balance_loss_coeff == 1.0
    for row in rows:
        assert row["knob"] == "moe_loss_fn.balance_loss_coeff"
        assert row["before"] == 1.0
        assert row["after"] == 4.0
        assert row["module_attr_untouched"] == 1.0


def test_tune_raises_when_no_target_module_exists() -> None:
    with pytest.raises(RuntimeError, match="no module exposes"):
        tune_balance_coeff(_Model([_Block("model.0", with_loss_fn=False)]), 4.0)


def test_read_balance_coeffs_skips_modules_without_a_loss_fn() -> None:
    model = _Model([_Block("model.5"), _Block("model.6", with_loss_fn=False)])
    assert read_balance_coeffs(model) == {"model.5": 1.0}


def test_assert_balance_coeff_detects_drift() -> None:
    model = _Model([_Block("model.5")])
    tune_balance_coeff(model, 4.0)
    assert assert_balance_coeff(model, 4.0)["ok"] is True

    model._children[0].moe_loss_fn.balance_loss_coeff = 1.0
    with pytest.raises(RuntimeError, match="drifted off 4.0"):
        assert_balance_coeff(model, 4.0)


def test_assert_balance_coeff_raises_without_targets() -> None:
    with pytest.raises(RuntimeError, match="no module exposes"):
        assert_balance_coeff(_Model([_Block("model.0", with_loss_fn=False)]), 4.0)
