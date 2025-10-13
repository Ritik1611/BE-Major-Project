# trainer_agent/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math
from typing import Tuple, Optional


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128), output_dim=2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, bottleneck=64, hidden=(256, 128)):
        super().__init__()
        enc_layers = []
        prev = input_dim
        for h in hidden:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(nn.ReLU())
            prev = h
        enc_layers.append(nn.Linear(prev, bottleneck))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        prev = bottleneck
        for h in reversed(hidden):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(nn.ReLU())
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def embed(self, x):
        return self.encoder(x)


def _to_device(tensor_or_model, device):
    if isinstance(tensor_or_model, torch.nn.Module):
        return tensor_or_model.to(device)
    return tensor_or_model.to(device)


def train_model(
    X: torch.Tensor,
    y: torch.Tensor,
    input_dim: int,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    model_type: str = "mlp",
    rag_k: int = 3,
    augment_with_neighbors: bool = False,
    X_reference: Optional[torch.Tensor] = None
) -> Tuple[dict, Optional[torch.nn.Module]]:
    """
    Trains either supervised model (if y has >=2 classes) or autoencoder (unsupervised)
    Returns: (delta_state_dict, model_instance)
    delta_state_dict is a mapping of tensors (trained_state - initial_state)
    model_instance is the trained PyTorch model (can be None in edge cases)
    """

    device = torch.device(device)
    X = X.to(device)

    # Decide whether supervised or unsupervised:
    supervised = False
    try:
        unique = torch.unique(y).numel()
        supervised = (unique >= 2)
    except Exception:
        supervised = False

    if supervised:
        # Decide classification or regression
        is_classification = (y.dtype == torch.long) or (len(torch.unique(y)) <= 20 and y.dtype != torch.float32)

        # Build dataset; if rag augmentation requested, X should already be augmented by caller
        model = SimpleMLP(input_dim, hidden_dims=(256, 128), output_dim=(len(torch.unique(y)) if is_classification else 1))
        model = model.to(device)

        loss_fn = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()
        opt = optim.Adam(model.parameters(), lr=lr)

        dataset = TensorDataset(X, y.to(device))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        init_state = {k: v.clone().detach() for k, v in model.state_dict().items()}

        for epoch in range(max(1, epochs)):
            model.train()
            running = 0.0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                out = model(xb)
                if is_classification:
                    # out shape (B,C)
                    loss = loss_fn(out, yb.long())
                else:
                    out = out.view(-1)
                    loss = loss_fn(out, yb.float())
                loss.backward()
                opt.step()
                running += float(loss.item())
            # print small progress-less verbose
        final_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
        # delta = final - init
        delta = {}
        for k in final_state.keys():
            delta[k] = (final_state[k].cpu() - init_state[k].cpu())
        return delta, model

    else:
        # Unsupervised: train autoencoder to reconstruct X; return encoder delta & model
        ae = AutoEncoder(input_dim, bottleneck=min(64, max(8, input_dim // 8)), hidden=(256, 128))
        ae = ae.to(device)
        opt = optim.Adam(ae.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        init_state = {k: v.clone().detach() for k, v in ae.state_dict().items()}
        for epoch in range(max(1, epochs)):
            ae.train()
            for (xb,) in loader:
                xb = xb.to(device)
                opt.zero_grad()
                recon = ae(xb)
                loss = loss_fn(recon, xb)
                loss.backward()
                opt.step()

        final_state = {k: v.clone().detach() for k, v in ae.state_dict().items()}
        # delta = final - init
        delta = {}
        for k in final_state.keys():
            delta[k] = (final_state[k].cpu() - init_state[k].cpu())
        # return the full AE model (so caller can attempt apply_delta_to_model)
        return delta, ae
