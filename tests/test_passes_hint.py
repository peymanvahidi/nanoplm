from nanoplm.pretraining.utils import format_passes_hint


class TestPassesHint:
    def test_exact_when_total_known(self):
        msg = format_passes_hint(budget_tokens=10_000_000, total_train_tokens=9_000_000)
        assert "10,000,000 tokens" in msg
        assert "1.1" in msg            # 10M / 9M ≈ 1.11 passes
        assert "approx" not in msg

    def test_approx_when_total_missing(self):
        msg = format_passes_hint(budget_tokens=10_000_000, total_train_tokens=0,
                                 approx_total=8_000_000)
        assert "approx" in msg
