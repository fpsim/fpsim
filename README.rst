Family planning model and analyses
==================================

This repository contains the code for both the family planning model, FPsim, as well as the analysis scripts for performing analyses.

* FPsim, in the folder ``fpsim``, is a standalone Python library for performing family planning analyses.
* Scripts that use FPsim to run analyses are in the ``fp_analyses`` folder.
* FPsim tests are in the ``tests`` folder.
* Most other folders are preserved for archival purposes and may be deleted.
* This repository is currently under development. Eventually, FPsim and ``fp_analyses`` will be split into a separate repositories.

Installation
------------

Run ``pip install -e .`` to install these packages and their required dependencies. This will make both ``fpsim`` and ``fp_analyses`` available on the Python path.