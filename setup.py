import io
import re
from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    'numpy>=1.15.4',
    'shapely~=1.7',
    'rtree>=0.8.3',
    'six'
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
    extras_require={'tests': ['pytest',
                              'pytest-cov'],
                    'tests_visualisation_utility': ['descartes'],
                    'docs': ['sphinx',
                             'sphinx-rtd-theme',
                             'recommonmark']},
    zip_safe=False)
