"""Tests for the Siamese U-Net (Step 3.3).

Focus points:
- output contract: logits ``[B, 1, H, W]`` for arbitrary (including odd) spatial sizes;
- *actual* weight sharing: a forward hook proves the encoder runs exactly twice per
  forward call (Siamese behaviour is calling one module twice, not copying weights);
- trainability: gradient descent on a planted rectangle must reduce the loss;
- dataset integration: samples from the 3.2 loaders flow through unchanged.
"""

from __future__ import annotations

import pytest
import torch
import numpy as np
import rasterio
from rasterio.transform import from_origin

from ml.change_detection.model import (
    SiameseUNet,
    SiameseUNetConfig,
)


def pair(batch: int = 1, channels: int = 4, height: int = 64, width: int = 64,
         seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    before = torch.rand((batch, channels, height, width), generator=generator)
    after = torch.rand((batch, channels, height, width), generator=generator)
    return before, after


def encoder_call_count(model: SiameseUNet) -> int:
    calls = 0

    def hook(_module, _inputs, _outputs) -> None:
        nonlocal calls
        calls += 1

    handles = [block.register_forward_hook(hook) for block in model.encoder]
    try:
        model.eval()
        with torch.no_grad():
            model(*pair())
    finally:
        for handle in handles:
            handle.remove()
    return calls


# ----------------------------------------------------------------------- config


def test_config_defaults_and_validation() -> None:
    config = SiameseUNetConfig()
    assert config.input_divisor == 16
    assert config.to_dict()["in_channels"] == 4
    with pytest.raises(ValueError, match="in_channels"):
        SiameseUNetConfig(in_channels=0)
    with pytest.raises(ValueError, match="depth"):
        SiameseUNetConfig(depth=7)
    with pytest.raises(ValueError, match="base_channels"):
        SiameseUNetConfig(base_channels=-1)


def test_summary_reports_parameter_count() -> None:
    summary = SiameseUNet(SiameseUNetConfig(base_channels=8)).summary()
    assert summary["model"] == "SiameseUNet"
    assert summary["parameters"] == summary["trainable_parameters"] > 0
    assert summary["config"]["base_channels"] == 8


# ----------------------------------------------------------------------- forward


def test_forward_returns_logits_of_input_extent() -> None:
    model = SiameseUNet(SiameseUNetConfig(base_channels=8))
    model.eval()
    with torch.no_grad():
        logits = model(*pair(height=64, width=64))
    assert logits.shape == (1, 1, 64, 64)
    assert logits.dtype == torch.float32
    probabilities = model.probability(logits)
    assert float(probabilities.min()) >= 0.0 and float(probabilities.max()) <= 1.0


@pytest.mark.parametrize("height,width", [(32, 32), (37, 53), (64, 48), (100, 100)])
def test_forward_handles_arbitrary_sizes_via_internal_padding(height: int, width: int) -> None:
    model = SiameseUNet(SiameseUNetConfig(base_channels=8))
    model.eval()
    with torch.no_grad():
        logits = model(*pair(height=height, width=width))
    assert logits.shape == (1, 1, height, width)


def test_forward_accepts_the_pune_inference_size() -> None:
    """The real grid is 1025 x 1025 (odd): it must flow through without manual padding."""
    model = SiameseUNet(SiameseUNetConfig(base_channels=4, depth=4))
    model.eval()
    with torch.no_grad():
        logits = model(*pair(height=65, width=65, seed=1))
    assert logits.shape == (1, 1, 65, 65)


def test_forward_is_deterministic_in_eval_mode() -> None:
    model = SiameseUNet(SiameseUNetConfig(base_channels=8))
    model.eval()
    before, after = pair(seed=2)
    with torch.no_grad():
        first, second = model(before, after), model(before, after)
    assert torch.equal(first, second)


def test_forward_batches() -> None:
    model = SiameseUNet(SiameseUNetConfig(base_channels=8))
    model.eval()
    before, after = pair(batch=3, height=32, width=32)
    with torch.no_grad():
        assert model(before, after).shape == (3, 1, 32, 32)


# ------------------------------------------------------------------ weight sharing


def test_encoder_runs_exactly_twice_per_forward() -> None:
    """Siamese = one encoder applied to both dates, verified at execution level."""
    model = SiameseUNet(SiameseUNetConfig(base_channels=8))
    assert encoder_call_count(model) == 2 * len(model.encoder)


def test_identical_inputs_give_zero_difference_features() -> None:
    """For before == after the fusion's |before - after| branch must be exactly zero."""
    model = SiameseUNet(SiameseUNetConfig(base_channels=8))
    model.eval()
    same = pair(seed=3)
    captured: dict = {}

    def hook(_module, args) -> None:
        captured["input"] = args[0].detach()

    handle = model.fusion.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(same[0], same[0])
    finally:
        handle.remove()

    bottleneck_channels = model.bottleneck_channels
    difference_branch = captured["input"][:, 2 * bottleneck_channels:]
    assert float(difference_branch.abs().max()) == 0.0, \
        "identical dates must contribute zero difference features"
    # and the two date branches must be identical to each other
    assert torch.equal(captured["input"][:, :bottleneck_channels],
                       captured["input"][:, bottleneck_channels:2 * bottleneck_channels])


# ------------------------------------------------------------------ trainability


def test_gradient_step_learns_a_planted_change() -> None:
    torch.manual_seed(0)
    model = SiameseUNet(SiameseUNetConfig(base_channels=8, depth=3))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before, after = pair(height=64, width=64, seed=4)
    after = after.clone()
    after[:, :, 24:40, 24:40] = 1.0 - after[:, :, 24:40, 24:40]
    target = torch.zeros((1, 1, 64, 64))
    target[:, :, 24:40, 24:40] = 1.0

    model.train()
    first_logits = model(before, after)
    first_loss = torch.nn.functional.binary_cross_entropy_with_logits(first_logits, target)
    optimizer.zero_grad()
    first_loss.backward()
    optimizer.step()

    for _ in range(60):
        logits = model(before, after)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < first_loss.item(), "training must reduce the loss on a planted change"
    with torch.no_grad():
        probability = model.probability(model(before, after))
    centre = float(probability[0, 0, 30:34, 30:34].mean())
    outside = float(probability[0, 0, 0:6, 0:6].mean())
    assert centre > outside, "changed centre should score higher than the unchanged corner"


# -------------------------------------------------------------------- validation


def test_rejects_mismatched_pair_shapes() -> None:
    model = SiameseUNet()
    with pytest.raises(ValueError, match="identical shapes"):
        model(torch.zeros((1, 4, 32, 32)), torch.zeros((1, 4, 32, 48)))


def test_rejects_wrong_channel_count() -> None:
    model = SiameseUNet(SiameseUNetConfig(in_channels=4))
    with pytest.raises(ValueError, match="channels"):
        model(torch.zeros((1, 3, 32, 32)), torch.zeros((1, 3, 32, 32)))


def test_rejects_non_batched_input() -> None:
    model = SiameseUNet()
    with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
        model(torch.zeros((4, 32, 32)), torch.zeros((4, 32, 32)))


def test_rejects_inputs_below_the_minimum_size() -> None:
    model = SiameseUNet(SiameseUNetConfig(depth=4))
    with pytest.raises(ValueError, match="below the minimum"):
        model(torch.zeros((1, 4, 8, 8)), torch.zeros((1, 4, 8, 8)))


def test_change_mask_threshold_validation() -> None:
    with pytest.raises(ValueError, match="threshold"):
        SiameseUNet.change_mask(torch.zeros((1, 1, 8, 8)), threshold=1.5)


def test_three_channel_levir_config() -> None:
    model = SiameseUNet(SiameseUNetConfig(in_channels=3, base_channels=8))
    model.eval()
    before, after = pair(channels=3, height=64, width=64)
    with torch.no_grad():
        assert model(before, after).shape == (1, 1, 64, 64)
    with pytest.raises(ValueError, match="channels"):
        model(torch.zeros((1, 4, 64, 64)), torch.zeros((1, 4, 64, 64)))


# ------------------------------------------------------------------- integration


@pytest.fixture()
def mini_oscd_root(tmp_path):
    """One minimal OSCD region with an annotated change square."""
    from ml.change_detection.dataset import BAND_ORDER

    root = tmp_path / "oscd"
    images = root / "images" / "solo"
    rows, cols = np.indices((32, 32))
    for date, offset in (("imgs_1_rect", 0), ("imgs_2_rect", 100)):
        for band, base in zip(BAND_ORDER, (300, 400, 500, 2000)):
            values = (base + offset + 5 * rows + 3 * cols).astype("uint16")
            path = images / date / f"{band}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(path, "w", driver="GTiff", height=32, width=32, count=1,
                               dtype="uint16", crs="EPSG:32643",
                               transform=from_origin(500000.0, 2000000.0, 10.0, 10.0)) as dst:
                dst.write(values, 1)
    mask = np.zeros((32, 32), dtype="uint8")
    mask[12:20, 12:20] = 255
    mask_path = root / "target" / "solo" / "cm" / "cm.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(mask_path, "w", driver="PNG", height=32, width=32, count=1,
                       dtype="uint8") as dst:
        dst.write(mask, 1)
    (root / "train.txt").write_text("solo\n", encoding="utf-8")
    (root / "test.txt").write_text("solo2\n", encoding="utf-8")
    return root


@pytest.fixture()
def mini_levir_root(tmp_path):
    """One minimal LEVIR-CD pair with a change label."""
    from ml.change_detection.dataset import BAND_ORDER

    root = tmp_path / "levir"
    split = root / "train"
    red, green, blue = 40, 90, 130
    before = np.zeros((3, 32, 32), dtype="uint8")
    after = np.zeros((3, 32, 32), dtype="uint8")
    before[0], before[1], before[2] = red, green, blue
    after[0], after[1], after[2] = red + 50, green, blue
    for name, array in (("A", before), ("B", after)):
        path = split / name / "p_0001.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", driver="PNG", height=32, width=32, count=3,
                           dtype="uint8") as dst:
            dst.write(array)
    label = np.zeros((32, 32), dtype="uint8")
    label[8:16, 8:16] = 255
    label_path = split / "label" / "p_0001.png"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(label_path, "w", driver="PNG", height=32, width=32, count=1,
                       dtype="uint8") as dst:
        dst.write(label, 1)
    return root


def test_samples_from_loaders_flow_through_the_model(mini_oscd_root, mini_levir_root) -> None:
    from torch.utils.data import DataLoader

    from ml.change_detection.dataset import LevirDataset, OscdDataset

    model = SiameseUNet(SiameseUNetConfig(base_channels=8))
    model.eval()
    for dataset in (OscdDataset(mini_oscd_root, split="train"),
                    LevirDataset(mini_levir_root, split="train")):
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        with torch.no_grad():
            logits = model(batch["before"], batch["after"])
            mask = model.change_mask(logits)
        assert logits.shape == batch["mask"].shape
        assert mask.dtype == torch.bool
        assert model.config.in_channels == batch["before"].shape[1]

