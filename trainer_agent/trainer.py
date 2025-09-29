import torch
import torch.optim as optim
import torch.nn as nn
from trainer_agent.model import DummyModel
from typing import Tuple, Dict, Any

def compute_weight_delta(before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    delta = {}
    for k in before.keys():
        # keep deltas on CPU
        delta[k] = (after[k].cpu() - before[k].cpu())
    return delta


def train_model(
    X: torch.Tensor,
    y: torch.Tensor,
    input_dim: int,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42
) -> Tuple[Dict[str, torch.Tensor], nn.Module]:
    """
    Train a simple DummyModel and return (delta_state_dict, trained_model).

    - The returned model is moved to CPU and set to eval() for safe reuse later.
    - The delta is a state-dict-like mapping of CPU tensors (after - before).
    - model._init_params is attached so external code can re-instantiate if needed.
    """
    # reproducibility
    torch.manual_seed(seed)

    # instantiate model on device
    model = DummyModel(input_dim=input_dim).to(device)

    # save initial state (on CPU)
    before = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}

    dataset = torch.utils.data.TensorDataset(X.to(device), y.to(device))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            # dummy loss (MSE towards zero vector)
            loss = (out ** 2).mean()
            loss.backward()
            opt.step()

    # save final state (on CPU)
    after = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
    delta = compute_weight_delta(before, after)

    # prepare model for return: move to CPU and set eval
    try:
        model = model.to("cpu")
    except Exception:
        # if moving fails for some reason, keep as-is (but normally it will succeed)
        pass
    model.eval()

    # record constructor/init params to help re-instantiation if needed elsewhere
    try:
        model._init_params = {"input_dim": input_dim}
    except Exception:
        # ignore if model doesn't allow attribute assignment
        pass

    return delta, model
