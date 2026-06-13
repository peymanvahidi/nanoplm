import pytest
from nanoplm.pretraining.utils import tokens_to_steps


class TestTokensToSteps:
    def test_exact_multiple(self):
        assert tokens_to_steps(1_000_000, 250_000) == 4

    def test_ceil_overshoot(self):
        # 1,000,001 / 250,000 -> ceil -> 5 (never undershoots the budget)
        assert tokens_to_steps(1_000_001, 250_000) == 5

    def test_minimum_one_step(self):
        assert tokens_to_steps(1, 250_000) == 1

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            tokens_to_steps(0, 250_000)
        with pytest.raises(ValueError):
            tokens_to_steps(1000, 0)
