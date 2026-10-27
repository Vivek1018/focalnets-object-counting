import torch

from focalnet_counting.model import CountingModel, ReconstructionModel


def test_forward_shapes():
    image = torch.rand(2, 3, 128, 128)
    recon = ReconstructionModel(dim=64, depth=2, patch_size=16)
    count = CountingModel(dim=64, depth=2, patch_size=16)
    reconstructed = recon(image)
    density = count(image)
    assert reconstructed.shape == image.shape
    assert density.shape == (2, 1, 128, 128)
    assert count.predicted_count(image).shape == (2,)
