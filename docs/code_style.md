# Code Style

These rules apply to everything under `src/`, `app/`, `scripts/` and `tests/`. Notebooks are for exploration only. Any logic that matters gets moved into the package.

---

## 1. Tooling (enforced in CI and pre-commit)

| Tool | Purpose | Config |
|---|---|---|
| `ruff format` | Formatter (Black-compatible) | line length 100 |
| `ruff check` | Linter: `E,F,W,I,B,UP,N,SIM,PTH,RUF,NPY,PD` | `pyproject.toml` |
| `mypy --strict` on `src/` | Static typing | `disallow_untyped_defs = true` |
| `pytest` | Tests | see `testing.md` |
| `pre-commit` | Runs the above + `nbstripout` + `check-added-large-files --maxkb=1024` | `.pre-commit-config.yaml` |
| `uv` | Env + lockfile | `uv.lock` committed |

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
[tool.ruff.lint]
select = ["E","F","W","I","B","UP","N","SIM","PTH","RUF","NPY","PD"]
[tool.mypy]
python_version = "3.11"
strict = true
plugins = ["pydantic.mypy"]
```

`check-added-large-files` exists because the reference repo committed a 1.1 GB dataset and 127 MB of pickles/weights.

## 2. Naming

| Thing | Convention | Example |
|---|---|---|
| Modules | `snake_case.py`, nouns | `vocab.py`, `decode.py` |
| Classes | `PascalCase` | `Vocabulary`, `AttentionDecoder` |
| Functions | `snake_case`, verbs | `build_vocab`, `beam_search` |
| Constants | `UPPER_SNAKE` | `PAD_IDX = 0` |
| Tensors | Name + shape comment at creation | `feats  # (B, 49, 2048)` |
| Configs | `snake_case` keys, nested by concern | `model.decoder_dim` |

Shape suffix convention in comments: `B` batch, `T` time steps, `L` regions (49), `E` embed, `D` decoder, `A` attention, `V` vocab, `C` encoder channels.

## 3. Python rules

1. **Type hints everywhere** in `src/`. Use `torch.Tensor` and document shapes in the docstring.
2. **No hard-coded paths.** Every path comes from config or `CAPTIONER_*` env vars. Use `pathlib.Path`, never string concatenation (the reference used `"D:\\..." + "/" + ...`).
3. **No `pickle`** for artifacts. Use JSON (vocab, config), HDF5 (features), `safetensors` (release weights), and `torch.save(state_dict)` (checkpoints, loaded with `weights_only=True`).
4. **No bare `except:`.** Catch specific exceptions and re-raise as domain errors (`InvalidImageError`, `ModelNotLoadedError`).
5. **No `print`** in library code. Use `logging.getLogger(__name__)`. CLI output goes through `rich`.
6. **Pure functions for data transforms.** `clean_caption(str) -> str` has no I/O, which keeps it trivial to test.
7. Functions ≤ 50 lines, modules ≤ 400 lines. If a module grows past that, split it.
8. Use `@dataclass(frozen=True)` or a pydantic model for structured returns, e.g. `Caption(text, tokens, score, attention)`, not tuples.
9. Constants for special tokens live in one place (`captioner.data.vocab`). Never write the literals `"<start>"`, `0`, etc. elsewhere.

## 4. PyTorch rules

- `model.train()` / `model.eval()` set explicitly in the loop. Inference runs under `torch.inference_mode()`.
- Device comes from one place, `utils.device.get_device()`, and is passed in. Never call `.cuda()` directly.
- `nn.Module.__init__` only builds layers. Weight init goes in `_init_weights()`.
- No Python loops over the batch dimension. Loops over time steps are fine in the decoder.
- Masks are `bool` tensors. Padding index is `PAD_IDX` everywhere (`Embedding(padding_idx=PAD_IDX)`, `CrossEntropyLoss(ignore_index=PAD_IDX)`).
- Seeding happens through `utils.seed.seed_everything(seed)`, which also seeds DataLoader workers via `worker_init_fn`.
- Any shape-changing op on a non-trivial line gets a trailing shape comment:

```python
context = (alpha.unsqueeze(-1) * feats).sum(dim=1)  # (B, C)
```

## 5. Config rules

- One YAML per experiment, inheriting `base.yaml` (`defaults: [base]`).
- Every hyper-parameter mentioned in `architecture.md` §6 is a config key. No magic numbers in code.
- The resolved config is saved with each run and is the only input `predictor.py` needs besides weights and vocab.

## 6. Docstrings

Google style, required on all public functions/classes in `src/`:

```python
def beam_search(model: CaptionModel, feats: torch.Tensor, beam: int, max_len: int) -> list[Caption]:
    """Decode captions with beam search.

    Args:
        model: Any model implementing ``init_state`` and ``step``.
        feats: Encoder output, (B, L, C) or (B, C).
        beam: Beam width; 1 == greedy.
        max_len: Hard cap on generated tokens including <end>.

    Returns:
        One best ``Caption`` per image, specials removed.
    """
```

## 7. Git

- Branches: `feat/…`, `fix/…`, `exp/…` (experiments), `docs/…`.
- Conventional Commits: `feat(decoder): add doubly stochastic regulariser`.
- PR checklist: tests pass, ruff/mypy clean, new config keys documented, no files > 1 MB.
- `main` is always runnable. Experiments that don't pan out stay on `exp/` branches.

## 8. Notebooks

- Live in `notebooks/`, numbered (`01_eda.ipynb`).
- Import from `captioner`, don't redefine pipeline functions.
- Outputs stripped on commit (`nbstripout`), except the final EDA figures, which are exported to `reports/`.

## 9. Streamlit app

- `app/streamlit_app.py` contains layout and callbacks only. All ML lives in `captioner.inference`.
- Heavy objects come from `@st.cache_resource`. No module-level model loading.
- No `unsafe_allow_html=True` (the reference used it for CSS). Theming goes in `.streamlit/config.toml` (see `design_system.md`).
