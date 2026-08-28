#!/usr/bin/env bash
# One-shot environment setup for every assignment in this repo.
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Creating .venv (Python 3.12)"
uv venv --python 3.12 .venv

echo "==> Installing dependencies"
VIRTUAL_ENV=.venv uv pip install -r requirements.txt

echo "==> Downloading spaCy model"
.venv/bin/python -m spacy download en_core_web_sm

echo "==> Downloading NLTK data"
.venv/bin/python - <<'PY'
import nltk
for pkg in [
    "punkt", "punkt_tab",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    "universal_tagset", "stopwords", "wordnet", "brown",
]:
    nltk.download(pkg, quiet=True)
print("NLTK data ready")
PY

echo "==> Registering Jupyter kernel 'nlpsandbox'"
.venv/bin/python -m ipykernel install --user --name nlpsandbox --display-name "Python (NLPSandbox)"

echo
echo "Done. Launch with:  .venv/bin/jupyter lab"
