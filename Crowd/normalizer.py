from __future__ import annotations

import numpy as np
import torch


class StandardScaler:
    def __init__(self, mean: float, std: float, eps: float = 1e-6):
        self.mean = float(mean)
        self.std = float(std)
        self.eps = float(eps)
        self._mean_t = None
        self._std_t = None

    @classmethod
    def fit(cls, values: np.ndarray, eps: float = 1e-6) -> "StandardScaler":
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return cls(0.0, 1.0, eps=eps)
        mean = float(np.mean(finite))
        std = float(np.std(finite))
        if std < eps:
            std = 1.0
        return cls(mean, std, eps=eps)

    def to(self, device: torch.device) -> "StandardScaler":
        self._mean_t = torch.tensor(self.mean, device=device)
        self._std_t = torch.tensor(self.std, device=device)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean_t is None or self._std_t is None:
            raise RuntimeError("Scaler must be moved to device with .to(device) before use")
        return (x - self._mean_t) / (self._std_t + self.eps)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean_t is None or self._std_t is None:
            raise RuntimeError("Scaler must be moved to device with .to(device) before use")
        return x * (self._std_t + self.eps) + self._mean_t
