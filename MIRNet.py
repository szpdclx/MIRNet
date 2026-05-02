import math
from typing import Optional

import torch
from torch import nn


class MultiDimReservoir(nn.Module):
    def __init__(
        self,
        input_dim: int,
        reservoir_dim: int,
        spectral_radius: float = 0.9,
        ridge_alpha: float = 1e-2,
        connectivity: float = 0.2,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.reservoir_dim = int(reservoir_dim)
        self.ridge_alpha = float(ridge_alpha)

        w_in = 2.0 * torch.rand(self.input_dim, self.reservoir_dim) - 1.0
        self.register_buffer("w_in", w_in)

        w_res = torch.empty(self.input_dim, self.reservoir_dim, self.reservoir_dim)
        for channel in range(self.input_dim):
            w = torch.rand(self.reservoir_dim, self.reservoir_dim) - 0.5
            mask = torch.rand_like(w) < connectivity
            w = w * mask
            eig_max = torch.linalg.eigvals(w).abs().max().clamp_min(1e-12)
            w_res[channel] = w * (spectral_radius / eig_max)
        self.register_buffer("w_res", w_res)

    def _step(self, x_t: torch.Tensor, hidden: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        input_part = torch.einsum("bd,dr->bdr", x_t, self.w_in)
        transformed = torch.einsum("bdr,drn->bdn", hidden, self.w_res)
        transformed = transformed / math.sqrt(self.reservoir_dim)
        cross_part = torch.einsum("bik,bin->bkn", corr, transformed)
        return torch.tanh(input_part + cross_part)

    def _ridge_readout(self, states: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feat_dim = states.shape
        target = target.unsqueeze(2)

        xt = states.transpose(1, 2)
        xtx = xt @ states
        reg = torch.eye(feat_dim, device=states.device, dtype=states.dtype)
        xtx = xtx + reg.unsqueeze(0) * self.ridge_alpha
        xty = xt @ target

        return torch.linalg.solve(xtx, xty).squeeze(2)

    def forward(self, x: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_dim = x.shape
        if corr.dim() == 2:
            corr = corr.unsqueeze(0).expand(batch_size, -1, -1)

        hidden = torch.zeros(
            batch_size,
            input_dim,
            self.reservoir_dim,
            device=x.device,
            dtype=x.dtype,
        )

        states = []
        targets = []
        for t in range(seq_len):
            hidden = self._step(x[:, t], hidden, corr)
            if t < seq_len - 1:
                states.append(hidden.unsqueeze(1))
                targets.append(x[:, t + 1].unsqueeze(1))

        states = torch.cat(states, dim=1)
        targets = torch.cat(targets, dim=1)

        readouts = []
        for channel in range(input_dim):
            states_c = states[:, :, channel, :]
            target_c = targets[:, :, channel]
            readouts.append(self._ridge_readout(states_c, target_c).unsqueeze(1))

        features = torch.cat(readouts, dim=1)
        return features.reshape(batch_size, input_dim * self.reservoir_dim)


class MIRNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        reservoir_dim: int = 100,
        spectral_radius: float = 0.9,
        ridge_alpha: float = 1e-2,
        corr_offdiag_gain: float = 3.0,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.corr_offdiag_gain = float(corr_offdiag_gain)

        self.corr_param = nn.Parameter(torch.eye(self.input_dim))
        self.reservoir = MultiDimReservoir(
            input_dim=self.input_dim,
            reservoir_dim=reservoir_dim,
            spectral_radius=spectral_radius,
            ridge_alpha=ridge_alpha,
        )

        feature_dim = self.input_dim * reservoir_dim
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def corr_matrix(self) -> torch.Tensor:
        eye = torch.eye(self.input_dim, device=self.corr_param.device, dtype=self.corr_param.dtype)
        offdiag = torch.tanh(self.corr_offdiag_gain * self.corr_param) * (1.0 - eye)
        return offdiag + eye

    def extract_features(self, x: torch.Tensor, corr: Optional[torch.Tensor] = None) -> torch.Tensor:
        if corr is None:
            corr = self.corr_matrix()
        return self.reservoir(x, corr)

    def forward_with_features(self, x: torch.Tensor):
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits, features

    def channel_influence_loss(
        self,
        task_loss: torch.Tensor,
        features: torch.Tensor,
        topk_ratio: float = 0.25,
    ) -> torch.Tensor:
        grad_feat = torch.autograd.grad(
            task_loss,
            features,
            retain_graph=True,
            create_graph=False,
        )[0].detach()

        batch_size = features.shape[0]
        reservoir_dim = features.shape[1] // self.input_dim
        grad_feat = grad_feat.view(batch_size, self.input_dim, reservoir_dim)

        influence = torch.einsum("bdr,ber->bde", grad_feat, grad_feat).mean(dim=0)
        influence = influence / (influence.abs().max() + 1e-6)

        topk = max(1, int(self.input_dim * topk_ratio))
        _, idx = torch.topk(influence.abs(), topk, dim=1)
        mask = torch.zeros_like(influence)
        mask.scatter_(1, idx, 1.0)

        eye = torch.eye(self.input_dim, device=features.device, dtype=features.dtype)
        offdiag = self.corr_matrix() * (1.0 - eye)
        return (offdiag.abs() * (1.0 - mask)).mean()

    def training_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: Optional[nn.Module] = None,
        chinf_lambda: float = 0.0,
        topk_ratio: float = 0.25,
    ):
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        logits, features = self.forward_with_features(x)
        task_loss = criterion(logits, y)

        if chinf_lambda > 0.0:
            reg_loss = self.channel_influence_loss(task_loss, features, topk_ratio)
            loss = task_loss + chinf_lambda * reg_loss
        else:
            reg_loss = task_loss.new_tensor(0.0)
            loss = task_loss

        return loss, logits, task_loss, reg_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(x)
        return logits
