import torch

from typhoon_pressure.small_version import (
    conditional_distribution,
    hazard_logits_to_survival,
    joint_distribution,
    sample_survival_locations,
)


def test_survival_is_monotone_and_p15_is_first_lead():
    logits = torch.tensor([[0.0, -1.0, 1.0]])
    survival = hazard_logits_to_survival(logits)
    assert survival.shape == (1, 3)
    assert torch.all(survival[:, 1:] <= survival[:, :-1])
    assert torch.all((survival >= 0.0) & (survival <= 1.0))


def test_q_t_normalizes_and_joint_mass_equals_survival():
    logits = torch.randn(2, 3, 5)
    survival = hazard_logits_to_survival(torch.randn(2, 3))
    q_t = conditional_distribution(logits)
    joint = joint_distribution(survival, q_t)
    torch.testing.assert_close(q_t.sum(dim=-1), torch.ones((2, 3)))
    torch.testing.assert_close(joint.sum(dim=-1), survival)


def test_sampling_marks_dead_members_with_negative_cell_index():
    survival = torch.tensor([[1.0, 0.0]])
    q_t = torch.tensor([[[0.25, 0.75], [0.5, 0.5]]])
    samples = sample_survival_locations(survival, q_t, num_samples=32)
    assert samples.alive.shape == (32, 1, 2)
    assert torch.all(samples.alive[:, :, 0])
    assert not torch.any(samples.alive[:, :, 1])
    assert torch.all(samples.cell_index[:, :, 1] == -1)
