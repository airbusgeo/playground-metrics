#!/usr/bin/env bash

# -------------------------------------------------------------------------------------------------------------------- #
# Your test suit run script here :

apt-get update
apt-get install -y libspatialindex-dev
pytest --cov-report term-missing  --cov=map_metric_api tests/

# -------------------------------------------------------------------------------------------------------------------- #
