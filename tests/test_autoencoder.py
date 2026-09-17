import torch

from dronetrackingrl.encoder.autoencoder import MaskedCropAutoencoder, train_autoencoder


def test_encode_produces_expected_latent_shape():
    model = MaskedCropAutoencoder(crop_size=32, latent_dim=8)
    crops = torch.rand(4, 1, 32, 32)

    latent = model.encode(crops)

    assert latent.shape == (4, 8)


def test_training_reduces_reconstruction_loss():
    torch.manual_seed(0)
    model = MaskedCropAutoencoder(crop_size=32, latent_dim=8)
    crops = torch.rand(8, 1, 32, 32)

    losses = train_autoencoder(model, crops, epochs=20, lr=1e-2)

    assert losses[-1] < losses[0]
