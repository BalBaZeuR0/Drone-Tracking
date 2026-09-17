from typing import List

import torch
from torch import nn


class MaskedCropAutoencoder(nn.Module):
    def __init__(self, crop_size: int = 64, latent_dim: int = 16):
        super().__init__()
        reduced = crop_size // 4
        self.crop_size = crop_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * reduced * reduced, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16 * reduced * reduced),
            nn.ReLU(),
            nn.Unflatten(1, (16, reduced, reduced)),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))


def train_autoencoder(
    model: MaskedCropAutoencoder, crops: torch.Tensor, epochs: int = 5, lr: float = 1e-3
) -> List[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        reconstruction = model(crops)
        loss = loss_fn(reconstruction, crops)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses
