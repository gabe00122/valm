from valm.config import AdamWConfig, OptimizerConfig, WarmupCosineConfig
from valm.utils.optimizer import make_optimizer


def test_make_optimizer_uses_total_steps_as_schedule_steps(monkeypatch):
    schedule_calls = []
    optimizer_calls = []

    def fake_schedule(**kwargs):
        schedule_calls.append(kwargs)
        return "lr_schedule"

    def fake_adamw(**kwargs):
        return ("adamw", kwargs)

    def fake_multisteps(tx, every_k_schedule):
        return ("multisteps", tx, every_k_schedule)

    def fake_optimizer(**kwargs):
        optimizer_calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(
        "valm.utils.optimizer.optax.warmup_cosine_decay_schedule",
        fake_schedule,
    )
    monkeypatch.setattr("valm.utils.optimizer.optax.adamw", fake_adamw)
    monkeypatch.setattr("valm.utils.optimizer.optax.MultiSteps", fake_multisteps)
    monkeypatch.setattr("valm.utils.optimizer.nnx.Optimizer", fake_optimizer)

    opt_config = OptimizerConfig(
        opt=AdamWConfig(lr=0.1),
        schedule=WarmupCosineConfig(warmup_ratio=0.1),
    )

    make_optimizer(
        model=object(),
        opt_config=opt_config,
        total_steps=5000,
        gradient_accumulations=8,
        wrt=object(),
    )

    assert schedule_calls == [
        {
            "init_value": 0.0,
            "peak_value": 0.1,
            "warmup_steps": 500,
            "decay_steps": 5000,
        }
    ]
    assert optimizer_calls[0]["tx"][2] == 8
