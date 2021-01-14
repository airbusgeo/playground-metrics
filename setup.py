import io
import re
from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    'numpy>=1.15.4',
    'shapely~=1.7',
    'rtree>=0.8.3',
]

with io.open('playground_metrics/__init__.py', 'rt', encoding='utf8') as f:
    version = re.search(
        r'__version__ = \'(.*?)\'', f.read(), re.M).group(1)

setup(
    name='playground-metrics',
    version=str(version),
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'docs.*']),
    description='Playground mAP scoring python API',
    include_package_data=True,
    author='Airbus DS GEO',
    author_email='jeffaudi@gmail.com',
    license='MIT',
    install_requires=REQUIRED_PACKAGES,
    extras_require={'lint': ['nox',
                             'flake8',
                             'flake8-docstrings',
                             'pep8-naming',
                             'flake8-mutable',
                             'flake8-eradicate',
                             'flake8-comprehensions',
                             'flake8-import-order'],
                    'tests': ['nox',
                              'pytest',
                              'pytest-cov'],
                    'tests_visualisation_utility': ['descartes'],
                    'docs': ['sphinx',
                             'sphinx-rtd-theme']},
    zip_safe=False)
