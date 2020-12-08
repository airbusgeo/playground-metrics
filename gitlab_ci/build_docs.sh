#!/usr/bin/env bash

# -------------------------------------------------------------------------------------------------------------------- #
# Your documentation build script here :

apt-get update
apt-get install -y libspatialindex-dev
git config core.symlinks true
sphinx-versioning build -r master -b -B $(git tag | sort -V | grep -v "[abrc]" | tail -1) docs/source docs/build/html

# -------------------------------------------------------------------------------------------------------------------- #
