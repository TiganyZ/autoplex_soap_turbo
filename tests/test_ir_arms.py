"""Arm expansion, and the prefactor check that keeps arms comparable.

`arm_pairs` decides which estimators run. Its new job is to refuse a
combination whose arms would be weighted differently from one another --
`acf`/`xl` on w^2 through ir_nu_power, `fft` on w(1-exp(-x)) through
ir_fft.f90's own default. Those differ by 16x at 3400 cm^-1 and no fitted scale
absorbs the difference, so ranking such arms against each other measures the
keyword rather than the models.
"""
from __future__ import annotations

import pytest

from autoplex_soap_turbo.turbogap.ir import ESTIMATORS, arm_pairs


def arms(*specs: dict) -> list[dict]:
    return [{"name": f"a{i}", **s} for i, s in enumerate(specs)]


def test_default_estimators_expand():
    got = arm_pairs({})
    assert [a["name"] for a in got] == [n for n, *_ in ESTIMATORS]
    assert {a["bias_mode"] for a in got} == {"acf", "xl", "fft"}


def test_one_family_is_left_alone():
    """A run of one family is self-consistent whatever the correction is; only
    its absolute number is convention-dependent, and that is not this check."""
    only_fft = arms({"bias_mode": "fft"}, {"bias_mode": "fft"})
    assert len(arm_pairs({"arms": only_fft, "quantum_correction": "harmonic"})) == 2

    only_mad = arms({"bias_mode": "acf"}, {"bias_mode": "xl"})
    assert len(arm_pairs({"arms": only_mad, "quantum_correction": "harmonic"})) == 2


def test_mixed_families_are_refused_under_harmonic():
    mixed = arms({"bias_mode": "fft"}, {"bias_mode": "acf"})
    with pytest.raises(ValueError, match="not comparable"):
        arm_pairs({"arms": mixed, "quantum_correction": "harmonic"})


@pytest.mark.parametrize("correction", ["classical", "quadratic", "CLASSICAL"])
def test_mixed_families_are_allowed_on_w2(correction):
    mixed = arms({"bias_mode": "fft"}, {"bias_mode": "xl"})
    assert len(arm_pairs({"arms": mixed, "quantum_correction": correction})) == 2


def test_the_default_correction_permits_the_default_arms():
    """The shipped estimator list mixes families, so a config that sets nothing
    must still be runnable -- which is only true because the repository's
    default is w^2 and not ir_fft.f90's."""
    assert len(arm_pairs({})) == len(ESTIMATORS)


def test_a_bad_correction_is_refused_rather_than_passed_through():
    """'linear' and 'none' are real ir_fft settings and neither is w^2."""
    mixed = arms({"bias_mode": "fft"}, {"bias_mode": "acf"})
    for correction in ("linear", "none"):
        with pytest.raises(ValueError, match="not comparable"):
            arm_pairs({"arms": mixed, "quantum_correction": correction})
