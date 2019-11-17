import os
from setuptools import setup, find_packages

# Get version
versionfile = '__init__.py'
versionstr = '__version__ = '
cwd = os.path.abspath(os.path.dirname(__file__))
versionpath = os.path.join(cwd, 'pyemod', versionfile)
with open(versionpath, 'r') as f:
    lines = f.readlines()
    assert lines[0].startswith(versionstr)
    version = lines[0].split(versionstr)[1].rstrip()

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
    name="pyemod",
    version=version,
    author="Cliff Kerr, Benoit Raybaud, Jonathan Bloedow, Clinton Collins, Chris Lorton, Daniel Klein",
    author_email="ckerr@idmod.org",
    description="Python tools for using EMOD",
    keywords=["EMOD"],
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'sciris',
    ],
)