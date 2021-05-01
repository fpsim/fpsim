==========
What's new
==========

All notable changes to the codebase are documented in this file. Changes that may result in differences in model output, or are required in order to run an old parameter set with the current version, are flagged with the term "Regression information".

.. contents:: **Contents**
   :local:
   :depth: 1


Coming soon
-----------
- Converting model to array-based implementation, which should increase performance by a factor of roughly 100.
- Adding a ``calib.calibrate()`` method.


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