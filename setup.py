import os
from setuptools import setup, find_packages

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
    name="fp_analyses",
    version='0.1',
    author="Cliff Kerr, Daniel Klein",
    author_email="ckerr@idmod.org",
    description="FP analyses",
    keywords=["EMOD"],
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'sciris',
    ],
)