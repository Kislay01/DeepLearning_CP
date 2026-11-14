from __future__ import annotations

import json
from pathlib import Path

import pytest

from captioner.config import Config
from captioner.data.manifest import manifest_sha256
from captioner.data.prepare import build_vocab_from_manifest, load_prepared, prepare_data
from captioner.data.vocab import (
    END_IDX,
    PAD_IDX,
    START_IDX,
    UNK_IDX,
    VocabError,
    Vocabulary,
    build_vocab,
)


def _vocab(min_freq: int = 1, max_len: int = 8) -> Vocabulary:
    caps = [["a", "dog", "runs"], ["a", "dog", "sits"], ["a", "cat", "start", "end"]]
    return build_vocab(caps, min_freq=min_freq, max_len=max_len)


def test_regression_D3_reserved_special_tokens() -> None:
    v = _vocab()
    assert v["<pad>"] == PAD_IDX == 0
    assert v["<start>"] == START_IDX == 1
    assert v["<end>"] == END_IDX == 2
    assert v["<unk>"] == UNK_IDX == 3
    # Real words "start"/"end" are ordinary tokens, distinct from the specials.
    assert v["start"] > 3 and v["end"] > 3
    assert v["start"] != v["end"]


def test_min_freq_and_ordering() -> None:
    v = _vocab(min_freq=2)
    assert v.itos[4:] == ("a", "dog")  # a=3, dog=2; others below threshold
    assert v["cat"] == UNK_IDX
    assert v.freqs == {"a": 3, "dog": 2}
    assert v.n_words == 2 and len(v) == 6 and "dog" in v


def test_encode_decode_roundtrip_and_padding() -> None:
    v = _vocab(max_len=8)
    ids = v.encode(["a", "dog", "runs"])
    assert ids[:5] == [START_IDX, v["a"], v["dog"], v["runs"], END_IDX]
    assert ids[5:] == [PAD_IDX] * 3 and len(ids) == 8
    assert v.decode(ids) == ["a", "dog", "runs"]
    assert v.encode(["a"], pad=False) == [START_IDX, v["a"], END_IDX]
    assert v.decode([START_IDX, v["a"], UNK_IDX, END_IDX, v["dog"]]) == ["a"]


def test_overlong_caption_truncated_keeping_end() -> None:
    v = _vocab(max_len=5)
    ids = v.encode(["a", "dog", "runs", "a", "dog", "sits"])
    assert len(ids) == 5 and ids[0] == START_IDX and ids[-1] == END_IDX
    assert v.decode(ids) == ["a", "dog", "runs"]


def test_json_roundtrip_and_hash(tmp_path: Path) -> None:
    v = build_vocab([["a", "b"]], min_freq=1, max_len=6, sha256_manifest="abc")
    v.save(tmp_path / "vocab.json")
    w = Vocabulary.load(tmp_path / "vocab.json")
    assert w == v and w.sha256() == v.sha256()
    data = json.loads((tmp_path / "vocab.json").read_text())
    assert data["specials"] == {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
    assert data["built_from"] == "train" and data["sha256_manifest"] == "abc"


def test_load_rejects_bad_files(tmp_path: Path) -> None:
    p = tmp_path / "v.json"
    p.write_text("{not json")
    with pytest.raises(VocabError):
        Vocabulary.load(p)
    good = build_vocab([["a"]], 1, 5).to_dict()
    for bad in ({**good, "version": 9}, {**good, "specials": {"<pad>": 1}}):
        p.write_text(json.dumps(bad))
        with pytest.raises(VocabError):
            Vocabulary.load(p)
    with pytest.raises(VocabError):
        Vocabulary(itos=("x",), freqs={}, min_freq=1, max_len=5)
    with pytest.raises(VocabError):
        Vocabulary(
            itos=("<pad>", "<start>", "<end>", "<unk>", "a", "a"), freqs={}, min_freq=1, max_len=5
        )


def test_regression_D5_vocab_built_from_train_only(mini_cfg: Config) -> None:
    prepared = prepare_data(mini_cfg)
    df = prepared.manifest
    words = {
        s: {w for c in df.loc[df["split"] == s, "caption_clean"] for w in str(c).split()}
        for s in ("train", "val", "test")
    }
    held_out_only = (words["val"] | words["test"]) - words["train"]
    assert held_out_only, "fixture should contain words that never occur in train"
    for word in held_out_only:
        assert prepared.vocab[word] == UNK_IDX
    assert set(prepared.vocab.itos[4:]) == words["train"]
    assert prepared.vocab.built_from == "train"


def test_vocab_hash_tracks_manifest(mini_cfg: Config) -> None:
    prepared = prepare_data(mini_cfg)
    assert prepared.vocab.sha256_manifest == manifest_sha256(prepared.manifest)
    assert load_prepared(mini_cfg).vocab == prepared.vocab
    changed = prepared.manifest.copy()
    changed.loc[0, "caption_clean"] = "a changed caption"
    assert manifest_sha256(changed) != prepared.manifest_sha256
    v2 = build_vocab_from_manifest(changed, 1, 25)
    assert v2.sha256_manifest != prepared.vocab.sha256_manifest
    v2.save(mini_cfg.paths.vocab_path)
    with pytest.raises(VocabError, match="different manifest"):
        load_prepared(mini_cfg)
