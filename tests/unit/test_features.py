from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from captioner.config import Config
from captioner.data.dataset import (
    CaptionDataset,
    FeatureStore,
    ImageCaptionDataset,
    ImageFeatureDataset,
    build_caption_table,
    collate_captions,
    collate_images,
    open_feature_store,
)
from captioner.data.download import raw_layout
from captioner.data.manifest import list_images
from captioner.data.prepare import load_prepared
from captioner.data.vocab import END_IDX, PAD_IDX, START_IDX
from captioner.features.extract import (
    FeatureFileError,
    feature_paths,
    validate_feature_file,
)
from captioner.models.encoder import ResNetEncoder
from captioner.utils.seed import seed_everything, worker_init_fn


def test_encoder_shapes_and_freezing() -> None:
    enc = ResNetEncoder(weights=None)
    assert not any(p.requires_grad for p in enc.parameters())
    enc.train()
    assert not any(m.training for m in enc.body.modules())  # frozen BN stays in eval mode
    with torch.inference_mode():
        out = enc(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 49, 2048)
    assert ResNetEncoder.pool(out).shape == (2, 2048)


def test_encoder_layer4_finetune_unfreezes_only_layer4() -> None:
    enc = ResNetEncoder(weights=None, train_layer4=True)
    trainable = {n.split(".")[1] for n, p in enc.named_parameters() if p.requires_grad}
    assert trainable == {"7"}  # body[7] == layer4
    enc.train()
    assert enc.body[7].training and not enc.body[0].training


def test_feature_files_schema(prepared_mini: Config) -> None:
    names = list_images(raw_layout(prepared_mini).image_dir)
    paths = feature_paths(prepared_mini)
    with h5py.File(paths["pooled"], "r") as fh:
        assert fh["features"].shape == (20, 2048)
        assert fh["features"].dtype == np.float16
        assert list(fh["image_ids"].asstr()[:]) == [n.removesuffix(".jpg") for n in names]
        assert fh.attrs["encoder"] == "resnet50" and fh.attrs["input"] == 224
        assert fh.attrs["weights"] == "random"
    with h5py.File(paths["spatial"], "r") as fh:
        ds = fh["features"]
        assert ds.shape == (20, 49, 2048) and ds.chunks == (1, 49, 2048)
        assert ds.compression == "lzf"
        pooled = h5py.File(paths["pooled"], "r")["features"][3].astype(np.float32)
        np.testing.assert_allclose(ds[3].astype(np.float32).mean(0), pooled, atol=2e-2, rtol=1e-2)
    for kind, path in paths.items():
        validate_feature_file(path, names, kind)


def test_validate_feature_file_errors(prepared_mini: Config, tmp_path: Path) -> None:
    names = list_images(raw_layout(prepared_mini).image_dir)
    pooled = feature_paths(prepared_mini)["pooled"]
    with pytest.raises(FeatureFileError, match="not found"):
        validate_feature_file(tmp_path / "x.h5", names, "pooled")
    with pytest.raises(FeatureFileError, match="shape"):
        validate_feature_file(pooled, names[:-1], "pooled")
    with pytest.raises(FeatureFileError, match="different image list"):
        validate_feature_file(pooled, list(reversed(names)), "pooled")
    bad = tmp_path / "bad.h5"
    with h5py.File(bad, "w") as fh:
        fh["features"] = np.zeros((20, 2048), dtype=np.float32)
    with pytest.raises(FeatureFileError, match="dtype"):
        validate_feature_file(bad, names, "pooled")


def test_caption_table_and_datasets(prepared_mini: Config) -> None:
    prep = load_prepared(prepared_mini)
    table = build_caption_table(prep.manifest, "train", prep.vocab)
    assert len(table) == 12
    assert table.captions.shape == (12, 5, prepared_mini.vocab.max_len)
    assert (table.captions[:, :, 0] == START_IDX).all()
    assert len(table.references[0]) == 5
    for img in range(12):
        for k in range(5):
            n = table.lengths[img, k]
            assert table.captions[img, k, n - 1] == END_IDX
            assert (table.captions[img, k, n:] == PAD_IDX).all()

    store = open_feature_store(feature_paths(prepared_mini)["spatial"], table)
    ds1 = CaptionDataset(table, store, captions_per_image=1)
    ds5 = CaptionDataset(table, store, captions_per_image=5)
    assert len(ds1) == 12 and len(ds5) == 60
    assert store._fh is None  # lazy: nothing opened until the first read
    feats, ids, length, idx = ds5[7]
    assert feats.shape == (49, 2048) and feats.dtype == torch.float32
    assert idx == 1 and ids[length - 1] == END_IDX
    assert store.__getstate__()["_fh"] is None
    store.close()

    imgs = ImageFeatureDataset(table, store)
    feats, i = imgs[0]
    assert feats.shape == (49, 2048) and i == 0
    batch_feats, batch_idx = collate_images([imgs[0], imgs[1]])
    assert batch_feats.shape == (2, 49, 2048) and batch_idx.tolist() == [0, 1]


def test_collate_sorts_and_trims(prepared_mini: Config) -> None:
    prep = load_prepared(prepared_mini)
    table = build_caption_table(prep.manifest, "train", prep.vocab)
    store = open_feature_store(feature_paths(prepared_mini)["pooled"], table)
    ds = CaptionDataset(table, store, captions_per_image=5)
    batch = collate_captions([ds[i] for i in range(8)])
    assert batch.feats.shape == (8, 2048)
    assert torch.all(batch.lengths[:-1] >= batch.lengths[1:])
    assert batch.captions.shape == (8, int(batch.lengths.max()))
    moved = batch.to(torch.device("cpu"))
    assert torch.equal(moved.captions, batch.captions)


def test_dataloader_with_workers_is_reproducible(prepared_mini: Config) -> None:
    prep = load_prepared(prepared_mini)
    table = build_caption_table(prep.manifest, "train", prep.vocab)
    store = open_feature_store(feature_paths(prepared_mini)["pooled"], table)

    def first_batch() -> torch.Tensor:
        gen = seed_everything(42)
        loader = DataLoader(
            CaptionDataset(table, store),
            batch_size=4,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_captions,
            worker_init_fn=worker_init_fn,
            generator=gen,
        )
        return next(iter(loader)).captions

    assert torch.equal(first_batch(), first_batch())


def test_feature_store_rejects_misaligned_ids(prepared_mini: Config) -> None:
    prep = load_prepared(prepared_mini)
    table = build_caption_table(prep.manifest, "val", prep.vocab)
    path = feature_paths(prepared_mini)["pooled"]
    with pytest.raises(ValueError, match="do not match"):
        FeatureStore(path, expected_ids=table.image_ids, index=table.feature_idx[::-1].copy())
    with pytest.raises(FileNotFoundError):
        open_feature_store(path.with_name("missing.h5"), table)
    with pytest.raises(ValueError, match="No rows"):
        build_caption_table(prep.manifest, "nope", prep.vocab)


def test_image_caption_dataset_for_finetuning(prepared_mini: Config) -> None:
    prep = load_prepared(prepared_mini)
    table = build_caption_table(prep.manifest, "test", prep.vocab)
    ds = ImageCaptionDataset(table, raw_layout(prepared_mini).image_dir)
    image, ids, length, idx = ds[0]
    assert image.shape == (3, 224, 224) and ids[0] == START_IDX and idx == 0 and length > 2
    assert (
        len(ImageCaptionDataset(table, raw_layout(prepared_mini).image_dir, captions_per_image=5))
        == 20
    )
    assert (
        ImageCaptionDataset(table, raw_layout(prepared_mini).image_dir, captions_per_image=5)[6][3]
        == 1
    )
