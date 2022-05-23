import os
import runpy
from setuptools import setup, find_packages

# Get version
cwd = os.path.abspath(os.path.dirname(__file__))
versionpath = os.path.join(cwd, 'fpsim', 'version.py')
version = runpy.run_path(versionpath)['__version__']

CLASSIFIERS = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GPLv3",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Development Status :: 1",
    "Programming Language :: Python :: 3.9",
]

setup(
    name="fpsim",
    version=version,
    author="Michelle O'Brien, Annie Valente, Cliff Kerr, Sam Buxton, Daniel Klein, Marita Zimmermann",
    author_email="info@fpsim.org",
    description="FPsim: Family Planning Simulator",
    keywords=["family planning", "women's health", "agent-based model", "simulation"],
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'numba',
        'scipy',
        'pandas>=1.3', 
        'sciris>=1.2.1',
        'matplotlib>=3.5.0',
        'seaborn>=0.9',
        'optuna',
        'plotnine',
        'pyarrow',
        'pyyaml',
    ],
)
