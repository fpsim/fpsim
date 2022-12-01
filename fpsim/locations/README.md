# Locations

This folder stores the location-specific data for FPsim. 

To add a new location:

1. Create a new folder with the (lowercase) location name `<name>`.
2. Add the source data files to the folder `<name>`.
3. Create a file `<name>.py` that contains the actual parameter values; see `senegal.py` for the correct structure.
4. Add `from . import <name>` to `__init__.py`.
5. In `fpsim/parameters.py`, add `elif location == '<name>': pars = fplocs.<name>.make_pars()` in the `pars()` function.