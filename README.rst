# Playground mAP scoring python API

[![python-version](http://playground-detection-scoring.theplayground-ml.appspot.com/badges/python.svg)](https://code.webfactory.intelligence-airbusds.com/innovation/machine-learning/playground-detection-scoring/) 
[![doc status](http://playground-detection-scoring.theplayground-ml.appspot.com/badges/pydoc.svg)](http://playground-detection-scoring.theplayground-ml.appspot.com/docs/) 
[![stable](http://playground-detection-scoring.theplayground-ml.appspot.com/badges/pystable.svg)](https://code.webfactory.intelligence-airbusds.com/innovation/machine-learning/playground-detection-scoring/tags) 
[![pipeline status](https://code.webfactory.intelligence-airbusds.com/innovation/machine-learning/playground-detection-scoring/badges/master/pipeline.svg)](https://code.webfactory.intelligence-airbusds.com/innovation/machine-learning/playground-detection-scoring/commits/master) 
[![coverage report](https://code.webfactory.intelligence-airbusds.com/innovation/machine-learning/playground-detection-scoring/badges/master/coverage.svg)](https://code.webfactory.intelligence-airbusds.com/innovation/machine-learning/playground-detection-scoring/commits/master)


[![latest](http://playground-detection-scoring.theplayground-ml.appspot.com/badges/pybuilddev3.6.svg)](https://code.webfactory.intelligence-airbusds.com/innovation/machine-learning/playground-detection-scoring/tags) 
[![latest](http://playground-detection-scoring.theplayground-ml.appspot.com/badges/pybuildtag3.6.svg)](https://code.webfactory.intelligence-airbusds.com/innovation/machine-learning/playground-detection-scoring/tags) 


[![test status27](http://playground-detection-scoring.theplayground-ml.appspot.com/badges/py2.7.svg)](https://code.webfactory.intelligence-airbusds.com/innovation/machine-learning/playground-detection-scoring/-/jobs) 
[![test status35](http://playground-detection-scoring.theplayground-ml.appspot.com/badges/py3.5.svg)](https://code.webfactory.intelligence-airbusds.com/innovation/machine-learning/playground-detection-scoring/-/jobs)
[![test status36](http://playground-detection-scoring.theplayground-ml.appspot.com/badges/py3.6.svg)](https://code.webfactory.intelligence-airbusds.com/innovation/machine-learning/playground-detection-scoring/-/jobs)
[![test status37](http://playground-detection-scoring.theplayground-ml.appspot.com/badges/py3.7.svg)](https://code.webfactory.intelligence-airbusds.com/innovation/machine-learning/playground-detection-scoring/-/jobs)


mAP stands for *mean average precision* and is a common metric in detection tasks and challenges.

In an effort to standardize evaluation tasks to implement efficient benchmarkings of algorithms, the following module
implements a generic python API to compute **mAP**, **AP** per label as well as **precision** and **recall** per label.

## Installation

Clone source, install dependencies and the package
```
sudo apt-get install libspatialindex-dev
git clone git@code.webfactory.intelligence-airbusds.com:innovation/machine-learning/playground-detection-scoring.git
cd playground-detection-scoring
pip install .
```

> Note: the rtree library depends on the `libspatialindex` C library which must be installed externally.

