==========
What's new
==========

All notable changes to the codebase are documented in this file. Changes that may result in differences in model output, or are required in order to run an old parameter set with the current version, are flagged with the term "Regression information".

.. contents:: **Contents**
   :local:
   :depth: 1


Version 0.6.0 (2021-05-05)
--------------------------
- Converted model to array-based implementation, increasing performance by a factor of roughly 100.
- Added regression and benchmarking tests.
- Added a ``calib.calibrate()`` method.