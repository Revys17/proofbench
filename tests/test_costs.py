import pytest

from src.costs import CostTracker, PRICING


class TestCostTracker:
    """Tests for token usage and cost tracking."""

    @pytest.mark.asyncio
    async def test_empty_tracker(self):
        tracker = CostTracker()
        s = await tracker.summary()
        assert s["total_cost_usd"] == 0.0
        assert s["per_model"] == {}

    @pytest.mark.asyncio
    async def test_single_record(self):
        tracker = CostTracker()
        await tracker.record("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)
        s = await tracker.summary()
        assert s["per_model"]["claude-sonnet-4-6"]["input_tokens"] == 1000
        assert s["per_model"]["claude-sonnet-4-6"]["output_tokens"] == 500
        assert s["per_model"]["claude-sonnet-4-6"]["api_calls"] == 1

    @pytest.mark.asyncio
    async def test_multiple_records_same_model(self):
        tracker = CostTracker()
        await tracker.record("claude-sonnet-4-6", 1000, 500)
        await tracker.record("claude-sonnet-4-6", 2000, 1000)
        s = await tracker.summary()
        info = s["per_model"]["claude-sonnet-4-6"]
        assert info["input_tokens"] == 3000
        assert info["output_tokens"] == 1500
        assert info["api_calls"] == 2

    @pytest.mark.asyncio
    async def test_multiple_models(self):
        tracker = CostTracker()
        await tracker.record("claude-sonnet-4-6", 1000, 500)
        await tracker.record("claude-haiku-4-5-20251001", 2000, 1000)
        s = await tracker.summary()
        assert len(s["per_model"]) == 2
        assert "claude-sonnet-4-6" in s["per_model"]
        assert "claude-haiku-4-5-20251001" in s["per_model"]

    @pytest.mark.asyncio
    async def test_cost_calculation(self):
        tracker = CostTracker()
        # Sonnet: $3/M input, $15/M output
        await tracker.record("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000)
        s = await tracker.summary()
        expected = 3.00 + 15.00  # $3 input + $15 output
        assert s["per_model"]["claude-sonnet-4-6"]["cost_usd"] == expected
        assert s["total_cost_usd"] == expected

    @pytest.mark.asyncio
    async def test_unknown_model_zero_cost(self):
        tracker = CostTracker()
        await tracker.record("unknown-model", 1000, 500)
        s = await tracker.summary()
        assert s["per_model"]["unknown-model"]["cost_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_total_cost_across_models(self):
        tracker = CostTracker()
        await tracker.record("claude-sonnet-4-6", 1_000_000, 0)       # $3.00
        await tracker.record("claude-haiku-4-5-20251001", 1_000_000, 0)  # $0.80
        s = await tracker.summary()
        assert s["total_cost_usd"] == pytest.approx(3.80, abs=0.01)

    @pytest.mark.asyncio
    async def test_total_cost_method(self):
        tracker = CostTracker()
        await tracker.record("claude-sonnet-4-6", 1_000_000, 1_000_000)
        assert await tracker.total_cost() == pytest.approx(18.0, abs=0.01)


class TestPricing:
    """Verify pricing table has expected entries."""

    def test_anthropic_models_present(self):
        assert "claude-opus-4-6" in PRICING
        assert "claude-sonnet-4-6" in PRICING
        assert "claude-haiku-4-5-20251001" in PRICING

    def test_pricing_format(self):
        for model, (inp, out) in PRICING.items():
            assert inp >= 0, f"{model} has negative input price"
            assert out >= 0, f"{model} has negative output price"
            assert out >= inp, f"{model} output should cost >= input"
