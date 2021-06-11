==========
What's new
==========

All notable changes to the codebase are documented in this file. Changes that may result in differences in model output, or are required in order to run an old parameter set with the current version, are flagged with the term "Regression information".

.. contents:: **Contents**
   :local:
   :depth: 1



Version 0.6.4 (2021-06-10)
--------------------------
- Added a ``MultiSim`` class, which can handle parallel runs and uncertainty bounds.
- *GitHub info*: PR `120 <https://github.com/amath-idm/fp_analyses/pull/120>`__


Version 0.6.3 (2021-06-10)
--------------------------
- Fixed a bug where exposure correction by age was accidentally being clipped to the range [0,1], restoring behavior of the array-based model to match the object-based model (notwithstanding stochastic effects and other bugfixes).
- *GitHub info*: PR `119 <https://github.com/amath-idm/fp_analyses/pull/119>`__


Version 0.6.2 (2021-05-10)
--------------------------
- Added ``fp.Intervention`` and ``fp.Analyzer`` classes, which are much more flexible ways to modify and record the state of the simulation, respectively.
- Fixed a bug with only females being born.
- *GitHub info*: PR `100 <https://github.com/amath-idm/fp_analyses/pull/100>`__


Version 0.6.1 (2021-05-02)
--------------------------
- Renamed ``fp.Calibration`` to ``fp.Experiment``, and added a new ``fp.Calibration`` class, using Optuna.
- This allows the user to do e.g. ``calib = fp.Calibration(pars); calib.calibrate(calib_pars)``
- Calibrating a single parameter takes about 20 seconds for a single parameter and a small population size (500 people). Realistic calibrations should take roughly 10 - 60 minutes.
- *GitHub info*: PR `93 <https://github.com/amath-idm/fp_analyses/pull/93>`__


Version 0.6.0 (2021-05-01)
--------------------------
- Refactored the model to use an array-based implementation, instead of a loop over individual people.
- This results in a performance increase of roughly 20-100x, depending on the size of the simulation. In practice, this means that 50,000 people can be run in roughly the same amount of time as 500 could be previously.
- *GitHub info*: PR `92 <https://github.com/amath-idm/fp_analyses/pull/92>`__


Version 0.5.2 (2021-04-30)
--------------------------
- Added a new script, ``preprocess_data.py``, that takes large raw data files and preprocesses them down to only the essentials used in the model.
- This increases the performance of ``calib.run()`` (**not** counting model runtime) by a factor of 1000.
- *GitHub info*: PR `91 <https://github.com/amath-idm/fp_analyses/pull/91>`__


Version 0.5.1 (2021-04-29)
--------------------------
- Added ``summarize()`` and ``to_json()`` methods to ``Calibration``. Also added an ``fp.diff_summaries()`` method for comparing them.
- Added regression and benchmarking tests (current total time: 24 s).
- Added a code coverage script (current code coverage: 59%).
- Added default flags for which quantities to compute.
- Split the logic of ``Calibration`` out into more detail: e.g., initialization, running, and post-processing.
- *GitHub info*: PR `90 <https://github.com/amath-idm/fp_analyses/pull/90>`__