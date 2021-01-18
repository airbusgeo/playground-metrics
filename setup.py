import io
import os
import re
from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    'numpy>=1.15.4',
    'shapely~=1.7',
    'rtree>=0.8.3',
]


base_dir = os.path.dirname(__file__)

with io.open(os.path.join(base_dir, 'playground_metrics/__init__.py'), 'rt', encoding='utf8') as f:
    version = re.search(
        r'__version__ = \'(.*?)\'', f.read(), re.M).group(1)


with open(os.path.join(base_dir, "README.rst")) as f:
    long_description = f.read()


with open(os.path.join(base_dir, "CHANGELOG.rst")) as f:
    changelog = f.read()
    long_description = "\n".join([long_description, changelog])


setup(
    name='playground-metrics',
    version=str(version),
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'docs.*']),
    description='Playground framework agnostic detection mAP scoring library.',
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
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    zip_safe=False)
