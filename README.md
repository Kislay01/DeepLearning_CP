# captioner — Image Caption Generator with Speech Output

CNN encoder (ResNet-50) + LSTM decoder with soft attention, trained on Flickr8k, served through an accessible Streamlit demo that reads the caption aloud.

> Work in progress. The full README (architecture, commands, results, limitations) lands with milestone M7. See `docs/STATUS.md`.

## Setup

```bash
make setup   # uv sync --frozen + pre-commit hooks (Python 3.11)
make lint test
```

## Licence

- Code: MIT (see `LICENSE`).
- Data: the Flickr8k images belong to their Flickr owners and are provided for **non-commercial research and educational use only**. Cite: M. Hodosh, P. Young and J. Hockenmaier (2013), "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics", *Journal of Artificial Intelligence Research* 47, 853–899. No dataset files are stored in this repository.
