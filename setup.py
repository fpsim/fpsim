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
    "Programming Language :: Python :: 3.7",
]

setup(
    name="fpsim",
    version=version,
    author="Annie Valente, Marita Zimmermann, Cliff Kerr, Daniel Klein",
    author_email="ckerr@idmod.org",
    description="FPsim and applications",
    keywords=["Family planning", "agent-based model", "simulation"],
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'numba',
        'scipy',
        'pandas',
        'sciris>=1.0',
        'matplotlib',
        'seaborn>=0.9',
        'optuna',
    ],
)
