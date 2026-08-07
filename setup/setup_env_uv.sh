#!/usr/bin/env bash

set -euo pipefail

env_name=autoplex_venv

ENV_DIR="${env_name}"

which uv || wget -qO- https://astral.sh/uv/install.sh | sh

source $HOME/.local/share/x86_64/../bin/env

is_uv_here=$?

if [ "$is_uv_here" == 0 ]; then
  echo "uv package is found"
else
  echo "uv is not found, installing by wget -qO- https://astral.sh/uv/install.sh | sh"
  wget -qO- https://astral.sh/uv/install.sh | sh
fi

uv python install 3.11

echo "> Creating virtual environment ($ENV_DIR) with Python 3.11"
uv venv "$ENV_DIR" --python 3.11

source "$ENV_DIR/bin/activate"

if [ -d "autoplex" ]; then
  cd autoplex
fi

uv pip install "scikit-build-core<0.10"
uv pip install "setuptools-scm>=8.0" nanobind

uv pip install --no-build-isolation phonopy==2.30.1

uv pip install -e ".[strict_all,dev,tests,docs]"

echo "Setup. To activate the environment run:"
echo "source $ENV_DIR/bin/activate"
