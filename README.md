# LLMs Encode Harmfulness and Refusal Separately — Reproduction

Reproducing experiments from [this paper](https://arxiv.org/abs/2507.11878). Code is organized with configs in `conf/`, experiment functions in `src/harm_refuse/experiments/`, and an entry point at `run.py`.

## Quick Start (uv)
- Set up environment with `uv sync`
- Hugging Face login for gated models/datasets: `huggingface-cli login`
- Run experiment: `uv run run.py -cn [EXPERIMENT]`

## Alternative (pip)
- Create and activate a Python 3.12 virtual env.
- Install this project and deps: `pip install -e .`
- (Optional) `huggingface-cli login`
- Run: `python run.py -cn [EXPERIMENT]`

## Notes

- Large models can be memory-heavy; use `smol` to validate setup.
