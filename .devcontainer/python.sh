#!/usr/bin/env bash
set -euo pipefail

uv pip install --system -e '.'
uv pip install --system -r requirements.txt
