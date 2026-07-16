"""Reference tests for the critic value representations (HL-Gauss and MSE).

The HL-Gauss head represents the value as a categorical distribution over
fixed bins; `value()` is the probability-weighted mean of the bin centers and
`loss()` is cross-entropy against a Gaussian smeared around the target
(Imani & White style histogram loss).
"""

import math

import numpy as np
from jax import numpy as jnp
from valm.config import HlGaussConfig
from valm.model.value_network import (
    HlGaussValueRepresentation,
    MseValueRepresentation,
    calculate_supports,
)


def _config(min=-2.0, max=2.0, n_logits=4, sigma=0.5):
    return HlGaussConfig(min=min, max=max, n_logits=n_logits, sigma=sigma)


def _norm_cdf(x, loc, scale):
    return 0.5 * (1.0 + math.erf((x - loc) / (scale * math.sqrt(2.0))))


def _reference_target_probs(target, config):
    """Gaussian probability mass per bin, renormalized to the support."""
    edges = np.linspace(config.min, config.max, config.n_logits + 1)
    target = min(max(target, config.min), config.max)
    cdf = np.array([_norm_cdf(e, target, config.sigma) for e in edges])
    z = cdf[-1] - cdf[0]
    return (cdf[1:] - cdf[:-1]) / z


def test_calculate_supports_bins_span_the_range():
    support, centers = calculate_supports(_config())

    np.testing.assert_allclose(np.asarray(support)[0], [-2.0, -1.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(np.asarray(centers), [-1.5, -0.5, 0.5, 1.5])


def test_uniform_logits_value_is_center_mean():
    config = _config()
    repr_ = HlGaussValueRepresentation(config, jnp.zeros((2, 3, config.n_logits)))

    value = np.asarray(repr_.value())

    # symmetric support -> mean of centers is 0
    np.testing.assert_allclose(value, np.zeros((2, 3)), atol=1e-6)


def test_peaked_logits_value_approaches_bin_center():
    config = _config()
    logits = np.full((1, 1, config.n_logits), -100.0, dtype=np.float32)
    logits[0, 0, 3] = 100.0  # all mass in the last bin

    value = np.asarray(HlGaussValueRepresentation(config, jnp.asarray(logits)).value())

    np.testing.assert_allclose(value, [[1.5]], atol=1e-5)


def test_hl_gauss_loss_matches_manual_cross_entropy():
    config = _config()
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(2, 3, config.n_logits)).astype(np.float32)
    targets = rng.uniform(-1.5, 1.5, size=(2, 3)).astype(np.float32)

    loss = np.asarray(
        HlGaussValueRepresentation(config, jnp.asarray(logits)).loss(
            jnp.asarray(targets)
        )
    )

    assert loss.shape == (2, 3)
    log_probs = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    for b in range(2):
        for t in range(3):
            target_probs = _reference_target_probs(float(targets[b, t]), config)
            expected = -(target_probs * log_probs[b, t]).sum()
            np.testing.assert_allclose(loss[b, t], expected, rtol=1e-4)


def test_hl_gauss_loss_is_minimized_by_the_target_distribution():
    """Cross-entropy is minimized when the predicted distribution equals the
    smeared target distribution."""
    config = _config()
    target = 0.7
    target_probs = _reference_target_probs(target, config)

    optimal_logits = jnp.asarray(np.log(target_probs)[None, None, :], jnp.float32)
    targets = jnp.full((1, 1), target)

    optimal = np.asarray(
        HlGaussValueRepresentation(config, optimal_logits).loss(targets)
    )

    rng = np.random.default_rng(1)
    for _ in range(5):
        other_logits = jnp.asarray(
            rng.normal(size=(1, 1, config.n_logits)), jnp.float32
        )
        other = np.asarray(
            HlGaussValueRepresentation(config, other_logits).loss(targets)
        )
        assert optimal[0, 0] <= other[0, 0] + 1e-6


def test_hl_gauss_out_of_range_targets_are_clipped():
    config = _config()
    logits = jnp.asarray(np.random.default_rng(2).normal(size=(1, 1, 4)), jnp.float32)
    repr_ = HlGaussValueRepresentation(config, logits)

    beyond = np.asarray(repr_.loss(jnp.full((1, 1), 100.0)))
    at_edge = np.asarray(repr_.loss(jnp.full((1, 1), config.max)))

    np.testing.assert_allclose(beyond, at_edge, rtol=1e-6)


def test_hl_gauss_getitem_slices_logits():
    config = _config()
    logits = jnp.asarray(
        np.random.default_rng(3).normal(size=(4, 6, config.n_logits)), jnp.float32
    )
    repr_ = HlGaussValueRepresentation(config, logits)

    sliced = repr_[:, :-1]

    np.testing.assert_allclose(
        np.asarray(sliced.value()), np.asarray(repr_.value())[:, :-1], rtol=1e-6
    )


def test_mse_representation_value_and_loss():
    values = jnp.asarray([[1.0, 2.0], [3.0, -1.0]])
    targets = jnp.asarray([[1.5, 2.0], [0.0, 1.0]])

    repr_ = MseValueRepresentation(values)

    np.testing.assert_allclose(np.asarray(repr_.value()), np.asarray(values))
    np.testing.assert_allclose(
        np.asarray(repr_.loss(targets)),
        0.5 * np.square(np.asarray(values) - np.asarray(targets)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(repr_[:, :1].value()), np.asarray(values)[:, :1]
    )
