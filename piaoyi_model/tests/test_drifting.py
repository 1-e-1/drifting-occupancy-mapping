import torch

from piaoyi_model.drift_loss import drifting_loss, second_order_drifting_loss


def test_drifting_loss_is_finite_and_differentiable():
    generated = torch.randn(2, 4, 16, requires_grad=True)
    positive = torch.randn(2, 5, 16)
    output = drifting_loss(generated, positive)
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()


def test_iterative_first_order_drifting_reports_displacement():
    generated = torch.randn(2, 4, 16, requires_grad=True)
    positive = torch.randn(2, 5, 16)
    output = drifting_loss(generated, positive, steps=12, step_size=1.0 / 12.0)
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.target_displacement)
    assert output.target_displacement > 0
    assert generated.grad is not None


def test_first_order_drifting_rejects_invalid_update_count():
    generated = torch.randn(1, 2, 4)
    positive = torch.randn(1, 2, 4)
    try:
        drifting_loss(generated, positive, steps=0)
    except ValueError as error:
        assert "at least one update" in str(error)
    else:
        raise AssertionError("steps=0 must be rejected")


def test_second_order_drifting_loss_is_finite_and_differentiable():
    generated = torch.randn(2, 4, 16, requires_grad=True)
    positive = torch.randn(2, 5, 16)
    output = second_order_drifting_loss(generated, positive, steps=4, alpha=3.0)
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()
    assert torch.isfinite(output.velocity_norm)
    assert torch.isfinite(output.target_displacement)
