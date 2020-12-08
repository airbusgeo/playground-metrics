#!/usr/bin/env bash

# -------------------------------------------------------------------------------------------------------------------- #
# Your lint script here :

apt-get update
apt-get install -y libspatialindex-dev
pip install flake8 radon pydocstyle
flake8 map_metric_api tests
#pydocstyle

# -------------------------------------------------------------------------------------------------------------------- #
