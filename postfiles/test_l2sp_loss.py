import pytest
import torch
from l2sp_loss import L2SPLoss


@pytest.fixture
def create_model_and_weights():
    """
    Creates a simple model and corresponding initial weights for testing L2SP loss.

    Returns:
        tuple: Contains a model and a list of its initial weights cloned for later comparisons.
    """
    torch.manual_seed(42)  # Ensure reproducibility
    model = torch.nn.Linear(10, 5)  # Simple linear model
    initial_weights = [param.data.clone() for param in model.parameters()]
    return model, initial_weights


def test_l2sp_basic(create_model_and_weights):
    """
    Test the basic functionality of L2SP loss to ensure it computes a positive loss
    when model parameters deviate from initial weights.
    """
    model, initial_weights = create_model_and_weights
    l2sp = L2SPLoss(initial_weights, l2sp_alpha=0.1)

    # Assume model parameters have changed during training
    for param in model.parameters():
        param.data += torch.randn_like(param.data)

    # Calculate L2SP Loss
    loss = l2sp(model)
    assert loss.item() > 0, "L2SP loss should be greater than zero for changed weights."
    print("Test L2SP Basic: Success - L2SP loss is correctly computed.")


def test_l2sp_zero_alpha(create_model_and_weights):
    """
    Test that L2SP loss returns zero when alpha is zero, indicating no regularization.
    """
    model, initial_weights = create_model_and_weights
    l2sp = L2SPLoss(initial_weights, l2sp_alpha=0.0)

    # Compute L2SP loss
    original_loss = l2sp(model)
    assert original_loss.item() == 0, "L2SP loss should be zero when alpha is zero."
    print(
        "Test L2SP Zero Alpha: Success - L2SP loss is zero as expected when alpha is zero."
    )


def test_l2sp_nonzero_alpha(create_model_and_weights):
    """
    Test L2SP loss with a non-zero alpha to ensure it applies a penalty correctly.
    """
    model, initial_weights = create_model_and_weights
    l2sp = L2SPLoss(initial_weights, l2sp_alpha=1.0)

    # Modify model parameters significantly
    for param in model.parameters():
        param.data += 5 * torch.ones_like(param.data)

    # Compute L2SP loss
    loss = l2sp(model)
    assert (
        loss.item() > 0
    ), "L2SP loss should be positive with nonzero alpha and changed parameters."
    print(
        "Test L2SP Nonzero Alpha: Success - L2SP loss"
        " applies penalty correctly with non-zero alpha."
    )


def test_l2sp_parameter_unchanged(create_model_and_weights):
    """
    Ensure computing the L2SP loss does not change the model parameters.
    """
    model, initial_weights = create_model_and_weights
    l2sp = L2SPLoss(initial_weights, l2sp_alpha=0.1)

    original_params = [p.clone() for p in model.parameters()]
    l2sp(model)  # Compute the loss but do not perform any update

    # Verify parameters remain unchanged
    for orig, new in zip(original_params, model.parameters()):
        assert torch.all(
            torch.eq(orig, new)
        ), "Model parameters should remain unchanged after computing L2SP loss."
    print(
        "Test L2SP Parameter Unchanged: Success"
        " - Model parameters remain unchanged after loss computation."
    )


def test_l2sp_all_zeros_initialization(create_model_and_weights):
    """
    Test L2SP loss behavior when initial weights are all zeros.
    """
    model, _ = create_model_and_weights
    zero_initial_weights = [torch.zeros_like(p) for p in model.parameters()]
    l2sp = L2SPLoss(zero_initial_weights, l2sp_alpha=0.1)

    # Compute L2SP loss
    loss = l2sp(model)
    assert (
        loss.item() > 0
    ), "L2SP loss should be positive when initial weights are all zeros\
        and model weights are non-zero."
    print(
        "Test L2SP All Zeros Initialization: Success - L2SP loss behaves correctly"
        " with all zeros initial weights."
    )
