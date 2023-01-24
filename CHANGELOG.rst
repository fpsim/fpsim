==========
What's new
==========

All notable changes to the codebase are documented in this file. Changes that may result in differences in model output, or are required in order to run an old parameter set with the current version, are flagged with the term "Regression information".

.. contents:: **Contents**
   :local:
   :depth: 1

Version 0.22.0 (2023-1-24)
--------------------------
- Add calibrate_manual.py to compare sim runs to data with new data structures
- Add plot_birth_spacing.py under senegal location to fine tune this calibration
- *GitHub info*: PR `https://github.com/fpsim/fpsim/pull/109>`_

Version 0.21.2 (2022-12-16)
---------------------------
- Updates Kenya, 2nd pass, completed 1st draft
- Starts calibrate_manual.py for Kenya with ASFR plot
- *GitHub info*: PR `76 <https://github.com/fpsim/fpsim/pull/76>`_

Version 0.21.1 (2022-12-09)
---------------------------
- Updates calibrated data to compare for Kenya, 1st pass
- Adds raw data to kenya folder
- *GitHub info*: PR `70 <https://github.com/fpsim/fpsim/pull/70>`_

Version 0.21.0 (2022-12-06)
---------------------------
- Updates contraceptive matrices in kenya.py to be from Kenya PMA 2019-2020
- Adds raw data to kenya folder and processing code to data_processing folder
- *GitHub info*: PR `51 <https://github.com/fpsim/fpsim/pull/51>`_


Version 0.20.0 (2022-11-30)
---------------------------
- Builds out new parameters file for Kenya
- Adds and reorganizes directories for external data files and data processing scripts
- *GitHub info*: PR `37 <https://github.com/fpsim/fpsim/pull/37>`_


Version 0.19.2 (2022-10-28)
---------------------------
- Added user guide
- *GitHub info*: PR `4 <https://github.com/fpsim/fpsim/pull/4>`_


Version 0.19.1 (2022-10-26)
---------------------------
- Moved to new repository location (http://github.com/fpsim/fpsim)
- Updated documentation in README
- Created new tutorials in tutorials folder
- Ordered tutorials by complexity through T1, T2, T3... Tn numbering system
- *GitHub info*: PR `1 <https://github.com/fpsim/fpsim/pull/1>`_


Version 0.19.0 (2022-09-01)
---------------------------
- Added age-specific plotting for tfr, pregnancies, imr, mmr, stillbirths, and births to Sim, MultiSim, and Scenarios
- Added ability to plot channels by age over the course of an interval of time (one year, for example)
- Added yearly age-specific plotting for pregnancies, imr and mmr
- *GitHub info*: PR `590 <https://github.com/amath-idm/fpsim/pull/590>`_


Version 0.18.2 (2022-08-12)
---------------------------
- Added age specific plotting for cpr, mcpr, and acpr to Sim, MultiSim, and Scenarios
- *GitHub info*: PR `584 <https://github.com/amath-idm/fpsim/pull/584>`_


Version 0.18.1 (2022-08-08)
---------------------------
- Added y-axis scaling to Sim.plot and MultiSim.plot()
- *GitHub info*: PR `583 <https://github.com/amath-idm/fpsim/pull/583>`_


Version 0.18.0 (2022-08-01)
---------------------------
- Adjusted stillbirth rates from Nori et al., which was conducted June 2022
- *GitHub info*: PR `560 <https://github.com/amath-idm/fpsim/pull/560>`_


Version 0.17.5 (2022-07-28)
---------------------------
- Refactored ExperimentVerbose and verbose_sim and related parts of test suite
- *GitHub info*: PR `471 <https://github.com/amath-idm/fpsim/pull/471>`_


Version 0.17.4 (2022-07-27)
---------------------------
- Added new test suite for the Scenarios API
- *GitHub info*: PR `527 <https://github.com/amath-idm/fpsim/pull/527>`_


Version 0.17.3 (2022-07-18)
---------------------------
- Added tutorial jupyter notebook to showcase Scenarios features
- *GitHub info*: PR `484 <https://github.com/amath-idm/fpsim/pull/484>`_


Version 0.17.2 (2022-07-15)
---------------------------
- Switched method mix plotting from line chart to stacked area chart for all classes
- *GitHub info*: PR `568 <https://github.com/amath-idm/fpsim/pull/568>`_


Version 0.17.1 (2022-07-14)
---------------------------
- Added example_scens.py for a quick debug of adding a novel method when developing new features
- Updated README with new debugging guidance
- GitHub info*: PR `570 <https://github.com/amath-idm/fpsim/pull/570>`_


Version 0.17.0 (2022-07-08)
---------------------------
- Added method mix timeseries plotting to Sim, MultiSim, and Scenarios through plot(to_plot='method')
- Added some test coverage for method mix plotting
- *GitHub info*: PR `554 <https://github.com/amath-idm/fpsim/pull/554>`_


Version 0.16.2 (2022-07-01)
---------------------------
- Refactors channel aggregation in Scenarios.analyze_sims()
- *GitHub info*: PR `561 <https://github.com/amath-idm/fpsim/pull/561>`_


Version 0.16.1 (2022-06-30)
---------------------------
- Add tracking of pregnancies
- Add cumulative sum of pregnancies to plotting functionality (see plot('apo'))
- *GitHub info*: PR `555 <https://github.com/amath-idm/fpsim/pull/555>`_


Version 0.16.0 (2022-06-28)
---------------------------
- Split matrix age category >25 into 26-35 and >35 
- Baseline contraceptive behavior remains the same, but interventions can differentiate now
- *GitHub info*: PR `551 <https://github.com/amath-idm/fpsim/pull/551>`_


Version 0.15.0 (2022-06-13)
---------------------------
- Added new plotting functionality ``Scenarios.plot('mortality')``
- Added new plotting functionality ``Scenarios.plot('apo')`` for adverse pregnancy outcomes
- Added ``stillbirths_over_year`` to keys, tracking, and plotting
- Added tracking of miscarriage, abortion, corresponding keys and plotting
- Temporarily commented out plot_interventions in ``sim.py`` to fix x-axis and vline issues in plotting
- *GitHub info*: PR `549 <https://github.com/amath-idm/fpsim/pull/549>`_


Version 0.14.2 (2022-06-06)
---------------------------
- Adding 3 new columns to the results dataframe in Scenarios


Version 0.14.2 (2022-05-31)
---------------------------
- Fixed bug in ``fp.snapshot()`` missing non-exact timesteps.
- Fixed bug with ``fp.timeseries_recorder()`` not being capable of being added as a kwarg.
- Tidied output of ``SimVerbose.story()``.
- Added ``sim.get_analyzer()`` and ``sim.get_intervention()`` methods (along with the plural versions).
- Renamed ``Experiment.dhs_data`` to ``Experiment.data``; likewise for ``model_to_calib`` → ``model``.
- Fixed bug with MCPR year plotting in ``Experiment``.
- Fixed bug with analyzers being applied only at the end of the sim instead of at every timestep.
- Fixed bug with interventions not plotting with simulations.
- Fixed bug with ``finalize()`` not being called for interventions.
- Increased code coverage of tests from 67% to 80%.
- *GitHub info*: PR `533 <https://github.com/amath-idm/fp_analyses/pull/533>`_


Version 0.14.1 (2022-05-27)
---------------------------
- Fixed bugs in how ``copy_from`` is implemented in scenarios.
- *GitHub info*: PR `526 <https://github.com/amath-idm/fp_analyses/pull/526>`_


Version 0.14.0 (2022-05-26)
---------------------------
- Adds an options module, allowing things like DPI to be set via ``fp.options(dpi=150)``.
- Updates plotting options and allows more control over style.
- Adds more control to plots, including ``start_year`` and ``end_year``.
- Adds a ``copy_from`` keyword to method probability update scenarios.
- Renames ``years`` to ``par_years`` in scenarios.
- Changes the logic of the ``People`` update step so that lactational amenorrhea is calculated after breastfeeding is updated.
- Changes the ``Sim`` representation to e.g. ``Sim("My sim"; n=10,000; 1960-2020; results: b=69,541 ☠=11,920 pop=62,630)``
- *GitHub info*: PR `522 <https://github.com/amath-idm/fp_analyses/pull/522>`__


Version 0.13.2 (2022-05-25)
---------------------------
- Added ASFR as an output of Experiments.
- ``MultiSim.run()`` now automatically labels un-labeled sims; this fixes bugs in MultiSim plotting functions.
- MultiSims also have additional error checking (e.g., they cannot be rerun).
- Refactored data files to be in "tall" instead of "wide" format.
- Removed years and age bins from summary statistics.
- *GitHub info*: PR `517 <https://github.com/amath-idm/fp_analyses/pull/517>`__


Version 0.13.1 (2022-05-25)
---------------------------
- Changed ``MultiSim.plot_method_mix()`` to be able to work with ``Scenarios``
- *GitHub info*: PR `513 <https://github.com/amath-idm/fp_analyses/pull/513>`__


Version 0.13.0 (2022-05-23)
---------------------------
- Changed parameters from a dictionary to a class and added ``parameters.py``. This class has additional validation, the ability to import from/export to JSON, etc.
- Restructured methods, including renaming ``pars['method_efficacy']`` to ``pars['methods']['eff']``, plus a new entry, ``pars['methods']['modern']``, to specify which are modern methods used for calculating MCPR.
- Methods have been reordered, grouping traditional and modern methods and sorting modern methods by longevity (e.g. condoms → pill → implants → IUDs).
- Added ability to add/remove contraceptive methods via ``pars.add_method()`` and ``pars.rm_method()``.
- Added a method to run a single scenario.
- *GitHub info*: PR `503 <https://github.com/amath-idm/fp_analyses/pull/503>`__


Version 0.12.0 (2022-05-22)
---------------------------
- Split FPsim repository from analyses scripts.
- Refactors ``experiment.py`` to load files for a specific location rather than being hard-coded.
- *GitHub info*: PR `504 <https://github.com/amath-idm/fp_analyses/pull/504>`__


Version 0.11.5 (2022-05-21)
---------------------------
- Improvements to the scenarios, including more helpful docstrings and error messages.
- Improved error checking of sims.
- *GitHub info*: PR `502 <https://github.com/amath-idm/fp_analyses/pull/502>`__


Version 0.11.4 (2022-05-20)
---------------------------
- Renamed parameter ``n`` to ``n_agents``, and adds parameter ``scaled_pop``.
- Tracking of switch events is disabled by default; set ``pars['track_switching'] = True`` to re-enable.
- Update default end year from 2019 to 2020.
- *GitHub info*: PR `496 <https://github.com/amath-idm/fp_analyses/pull/496>`__


Version 0.11.3 (2022-05-20)
---------------------------
- Tidied ``tests`` folder.
- Removed the calibration database by default (to keep, use ``fp.Calibration(keep_db=True)``.
- *GitHub info*: PR `495 <https://github.com/amath-idm/fp_analyses/pull/495>`__


Version 0.11.2 (2022-05-20)
---------------------------
- Added a ``people.make_pregnant()`` method.
- *GitHub info*: PR `494 <https://github.com/amath-idm/fp_analyses/pull/494>`__


Version 0.11.1 (2022-05-20)
---------------------------
- Replaced ``high`` and ``low`` breastfeeding duration parameters with Gumbel distribution parameters ``mu`` and ``beta``.
- *GitHub info*: PR `493 <https://github.com/amath-idm/fp_analyses/pull/493>`__


Version 0.11.0 (2022-05-20)
---------------------------
- Major refactor of ``senegal.py``, organizing parameters into groups and renaming.
- Parameter names made more consistent, e.g. ``exposure_correction`` → ``exposure_factor``, ``maternal_mortality_multiplier`` → ``maternal_mortality_factor``.
- Added comprehensive parameter checking.
- Updates to the default representation: ``print(sim)`` is now a very brief representation; use ``sim.disp()`` to get the old behavior.
- *GitHub info*: PR `492 <https://github.com/amath-idm/fp_analyses/pull/492>`__


Version 0.10.7 (2022-05-19)
---------------------------
- Updated ``fp.Scenarios()`` API.
- Added a new ``fp.Scenario()`` class, with a convenience function ``fp.make_scen()`` for creating new scenarios, for later use with ``fp.Scenarios()``.
- *GitHub info*: PR `488 <https://github.com/amath-idm/fp_analyses/pull/488>`__


Version 0.10.6 (2022-05-19)
---------------------------
- Adds ``fp.parallel()`` to quickly run multiple sims in parallel and return a ``MultiSim`` object.
- Adds an ``fp.change_par()`` intervention.
- *GitHub info*: PR `487 <https://github.com/amath-idm/fp_analyses/pull/487>`__


Version 0.10.5 (2022-05-18)
---------------------------
- Changes how the matrices are implemented. For example, ``sim['methods']['probs']['18-25']`` has been renamed ``sim['methods']['raw']['annual']['18-25']``; ``sim['methods']['probs']['18-25']`` has been renamed ``sim['methods']['adjusted']['annual']['18-25']``; ``sim['methods_postpartum']['probs1to6']['18-25']`` has been renamed ``sim['methods']['adjusted']['pp1to6']['18-25']``; etc.
- Various other parameters were renamed for consistency (e.g. ``years`` → ``year``).
- Various other methods were renamed for clarity (e.g. ``maternal_mortality()`` → ``check_maternal_mortality()``; ``check_mcpr()`` → ``track_mcpr()``).
- Input validation has been added to the ``Scenarios`` class.
- Fixed ``fp.update_methods()`` so it can no longer produce probabilities >1.
- Removed a circular import in ``scenarios.py``.
- *GitHub info*: PR `482 <https://github.com/amath-idm/fp_analyses/pull/482>`__


Version 0.10.4 (2022-05-17)
---------------------------
- Fixes bugs with the MCPR growth implementation, as well as the wrong matrix being used.
- Added three new parameters: ``mcpr_growth_rate``, ``mcpr_max``, and ``mcpr_norm_year``, to control how MCPR growth is projected into the future.
- Updated ``sim.run()`` to return ``self`` rather than ``self.results``.
- *GitHub info*: PR `480 <https://github.com/amath-idm/fp_analyses/pull/480>`__


Version 0.10.3 (2022-05-12)
---------------------------
- Move country-specific parameters from ``fpsim.data`` to ``fpsim.locations``.
- *GitHub info*: PR `464 <https://github.com/amath-idm/fp_analyses/pull/464>`__


Version 0.10.2 (2022-05-10)
---------------------------
- Refactored ``People.get_method()`` to use more efficient looping.
- Numbafied ``n_multinomial()`` to get a ~20% speed increase.
- Added a ``method_timestep`` parameter to allow skipping contraceptive matrix updates (saves significant time for small sims).
- Added ``fp.pars(location='test')`` to use defaults for testing (e.g. small population size).
- Fixed divide-by-zero bug for small population sizes in total fertility rate.
- Refactored tests; they should now run locally in ~15 s.
- *GitHub info*: PR `448 <https://github.com/amath-idm/fp_analyses/pull/448>`__


Version 0.10.1 (2022-05-09)
---------------------------
- Fix ``Scenarios`` class.
- *GitHub info*: PR `433 <https://github.com/amath-idm/fp_analyses/pull/433>`__


Version 0.10.0 (2022-05-08)
---------------------------
- Moved Senegal parameters into FPsim.
- Added age of sexual debut.
- *GitHub info*: PR `427 <https://github.com/amath-idm/fp_analyses/pull/427>`__


Version 0.9.0 (2022-05-05)
--------------------------
- Added a new ``Scenarios`` class.
- *GitHub info*: PR `416 <https://github.com/amath-idm/fp_analyses/pull/416>`__


Version 0.8.0 (2021-08-28)
--------------------------
- Refactored the ``People`` object to use a new filtering-based approach.
- *GitHub info*: PR `219 <https://github.com/amath-idm/fp_analyses/pull/219>`__


Version 0.7.3 (2021-07-15)
--------------------------
- Fix bug to ensure that at least one process runs on each worker.
- *GitHub info*: PR `163 <https://github.com/amath-idm/fp_analyses/pull/163>`__


Version 0.7.2 (2021-07-14)
--------------------------
- Allow ``total_trials`` to be passed to an ``fp.Calibration`` object.
- *GitHub info*: PR `162 <https://github.com/amath-idm/fp_analyses/pull/162>`__


Version 0.7.1 (2021-07-14)
--------------------------
- Allow ``weights`` to be passed to an ``fp.Calibration`` object.
- *GitHub info*: PR `161 <https://github.com/amath-idm/fp_analyses/pull/161>`__


Version 0.7.0 (2021-06-29)
--------------------------
- Added new calibration plotting methods.
- Separated Experiment and Calibration into separate files, and renamed ``model.py`` to ``sim.py``.
- Fixed a bug where the age pyramid was being unintentionally modified in-place.
- *GitHub info*: PR `144 <https://github.com/amath-idm/fp_analyses/pull/144>`__


Version 0.6.5 (2021-06-11)
--------------------------
- Added R support; see ``examples/example_sim.R``.
- Fixed a bug where the age pyramid was being unintentionally modified in-place.
- *GitHub info*: PR `128 <https://github.com/amath-idm/fp_analyses/pull/128>`__


Version 0.6.4 (2021-06-10)
--------------------------
- Added a ``MultiSim`` class, which can handle parallel runs and uncertainty bounds.
- *GitHub info*: PR `124 <https://github.com/amath-idm/fp_analyses/pull/124>`__


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
