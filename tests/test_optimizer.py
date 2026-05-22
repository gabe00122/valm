import pytest

from vaml.utils.optimizer import scheduled_optimizer_steps


def test_scheduled_optimizer_steps_uses_emitted_multistep_updates():
    assert scheduled_optimizer_steps(40000, 8) == 5000


def test_scheduled_optimizer_steps_without_multistep_uses_rollout_updates():
    assert scheduled_optimizer_steps(40000, None) == 40000


def test_scheduled_optimizer_steps_rejects_non_positive_multistep():
    with pytest.raises(ValueError):
        scheduled_optimizer_steps(40000, 0)
