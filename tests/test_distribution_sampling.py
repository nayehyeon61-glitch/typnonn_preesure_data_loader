import torch

from typhoon_pressure.small_version import (
    AdaptiveDistributionSampler,
    DistributionSamplingConfig,
    SmallModelConfig,
)
from typhoon_pressure.small_version.losses import sampled_distribution_cross_entropy
from typhoon_pressure.small_version.losses import storm_trajectory_gaussian_nll


def _sampler(num_samples=32, gpt_state_dim=3):
    model_config = SmallModelConfig(
        input_dim=4,
        hidden_dim=8,
        distribution_start_day=15,
        distribution_end_day=17,
        distribution_step_days=1,
        lat_bin_deg=90,
        lon_bin_deg=180,
    )
    sampling_config = DistributionSamplingConfig(
        num_samples=num_samples,
        min_process_std_deg=0.1,
        max_process_std_deg=2.0,
        max_daily_displacement_deg=5.0,
        grid_kernel_std_deg=20.0,
    )
    return AdaptiveDistributionSampler(
        model_config,
        model_dim=8,
        gpt_state_dim=gpt_state_dim,
        sampling_config=sampling_config,
    )


def test_adaptive_sampler_outputs_psd_covariance_and_normalized_distribution():
    torch.manual_seed(7)
    sampler = _sampler(num_samples=16)
    future = torch.randn(2, 3, 8, requires_grad=True)
    gpt = torch.randn(2, 3)
    gpt_mask = torch.ones_like(gpt)

    outputs = sampler(future, gpt, gpt_mask)

    assert outputs["distribution_mean_latlon"].shape == (2, 3, 2)
    assert outputs["distribution_samples"].shape == (2, 16, 3, 2)
    assert outputs["distribution_process_cholesky"].shape == (2, 3, 2, 2)
    assert outputs["distribution_process_covariance"].shape == (2, 3, 2, 2)
    assert outputs["distribution_probabilities"].shape == (2, 3, 4)
    assert outputs["distribution_log_probabilities"].shape == (2, 3, 4)
    assert torch.isfinite(outputs["distribution_log_probabilities"]).all()
    assert torch.allclose(
        outputs["distribution_probabilities"].sum(dim=-1),
        torch.ones(2, 3),
        atol=1e-5,
    )

    assert torch.count_nonzero(outputs["distribution_process_covariance"][:, 0]) == 0
    eigenvalues = torch.linalg.eigvalsh(outputs["distribution_process_covariance"][:, 1:])
    assert torch.all(eigenvalues > 0)
    assert torch.all(torch.linalg.eigvalsh(outputs["distribution_initial_covariance"]) > 0)

    target = torch.zeros(2, 3, 4)
    target[..., 0] = 0.4
    target[..., 1] = 0.6
    mask = torch.ones(2, 3)
    loss = sampled_distribution_cross_entropy(
        outputs["distribution_log_probabilities"], target, mask
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert sampler.process_noise_head.weight.grad is not None
    assert torch.count_nonzero(sampler.process_noise_head.weight.grad) > 0
    assert sampler.initial_covariance_head.weight.grad is not None


def test_samples_are_time_correlated_recursive_trajectories():
    torch.manual_seed(13)
    sampler = _sampler(num_samples=512, gpt_state_dim=0)
    with torch.no_grad():
        sampler.fallback_state_head.weight.zero_()
        sampler.fallback_state_head.bias.zero_()
        sampler.initial_residual_head.weight.zero_()
        sampler.initial_residual_head.bias.zero_()
        sampler.drift_head.weight.zero_()
        sampler.drift_head.bias.zero_()
    future = torch.zeros(1, 3, 8)
    samples = sampler(future)["distribution_samples"][0]

    # With zero learned drift, day t+1 = day t + newly sampled process noise.
    # Therefore members keep their identity through time and adjacent lead
    # positions are positively correlated; independent per-day sampling would
    # have correlation close to zero.
    first = samples[:, 0, 0]
    second = samples[:, 1, 0]
    correlation = torch.corrcoef(torch.stack((first, second)))[0, 1]
    assert correlation > 0.5


def test_missing_gpt_state_is_valid_for_noise_conditioning():
    torch.manual_seed(21)
    sampler = _sampler(num_samples=8)
    future = torch.randn(1, 3, 8)
    gpt = torch.randn(1, 3)
    missing = torch.zeros_like(gpt)
    outputs = sampler(future, gpt, missing)
    assert torch.isfinite(outputs["distribution_process_covariance"]).all()
    assert torch.isfinite(outputs["distribution_log_probabilities"]).all()


def test_day15_anchor_and_storm_nll_train_p15_q_and_drift():
    torch.manual_seed(31)
    sampler = _sampler(num_samples=8, gpt_state_dim=0)
    future = torch.randn(2, 3, 8, requires_grad=True)
    endpoint = torch.tensor([[18.0, 132.0], [22.0, 145.0]])
    outputs = sampler(future, weathernext_endpoint_latlon=endpoint, weathernext_endpoint_mask=torch.ones(2))
    assert torch.allclose(outputs["distribution_marginal_covariance"][:, 0], outputs["distribution_initial_covariance"])
    target = outputs["distribution_mean_latlon"].detach() + 0.5
    loss = storm_trajectory_gaussian_nll(outputs["distribution_mean_latlon"], outputs["distribution_marginal_covariance"], target, torch.ones(2, 3))
    loss.backward()
    assert torch.count_nonzero(sampler.initial_covariance_head.weight.grad) > 0
    assert torch.count_nonzero(sampler.process_noise_head.weight.grad) > 0
    assert torch.count_nonzero(sampler.drift_head.weight.grad) > 0
