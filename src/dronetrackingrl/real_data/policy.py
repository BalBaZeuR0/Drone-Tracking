"""Kameradan bağımsız (paylaşılan skorlayıcı) PPO politikası.

Düz bir MLP, gözlemdeki tüm kameraların latent'lerini yan yana görür ve
"şu kamerayı seç" alışkanlıkları öğrenebilir (eğitimde cam4 hep görünürse cam4'ü
seçmek gibi) -- bu, gözlemdeki bilgiyi kullanmak yerine kameraya özgü bir
önyargıdır ve başka aralıklarda çöker. Burada her kameranın latent'i AYNI küçük
ağdan geçirilip bir skor üretilir; skorlar doğrudan aksiyon logit'leridir. Ağ
kamera kimliğini bilmediği için "gözlemde drone görünüyorsa seç" kuralı tüm
kameralara aynı şekilde uygulanır (permütasyon-eşdeğer politika). Değer (critic)
tarafı kameralar üzerinde ortalama alır (permütasyon-değişmez).
"""

from typing import Any, Dict, Tuple

import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class SharedCameraExtractor(nn.Module):
    def __init__(self, num_cameras: int, latent_dim: int, hidden: int = 32):
        super().__init__()
        self.num_cameras = num_cameras
        self.latent_dim = latent_dim
        self.scorer = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.value_encoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.Tanh())
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.latent_dim_pi = num_cameras
        self.latent_dim_vf = hidden

    def _per_camera(self, features: torch.Tensor) -> torch.Tensor:
        return features.reshape(-1, self.num_cameras, self.latent_dim)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.scorer(self._per_camera(features)).squeeze(-1)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.value_encoder(self._per_camera(features)).mean(dim=1))

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class SharedCameraPolicy(ActorCriticPolicy):
    def __init__(self, *args, num_cameras: int, latent_dim: int, scorer_hidden: int = 32, **kwargs):
        self._num_cameras = num_cameras
        self._latent_dim = latent_dim
        self._scorer_hidden = scorer_hidden
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SharedCameraExtractor(self._num_cameras, self._latent_dim, self._scorer_hidden)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # Skorlar zaten logit: aksiyon katmanını birim matris yapıp dondur, böylece
        # kameralar arası karıştırma (=kameraya özgü önyargı) öğrenilemez.
        with torch.no_grad():
            self.action_net.weight.copy_(torch.eye(self._num_cameras))
            self.action_net.bias.zero_()
        self.action_net.requires_grad_(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            num_cameras=self._num_cameras, latent_dim=self._latent_dim, scorer_hidden=self._scorer_hidden
        )
        return data
