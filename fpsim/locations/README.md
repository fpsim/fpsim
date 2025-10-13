# Locations

This folder stores the location-specific files for FPsim models (both the `<model>.py` file as well as location-specific data). 

## Adding a Location to this Repo

### To add a country model:

1. Create a new folder with the (lowercase) location name `<name>`.
2. Create a folder `<name>/data` and add the source data files there (see the 'Model Parameters' section below for the specific files required for an FPsim model to run). Many of the data files have corresponding R processing scripts to format the data, and some are from specific studies and need to be formatted manually. For the proper format of each file, refer to the files in `locations/kenya/data`. **Note that the filenames MUST match the naming conventions defined in the table(s) below.**
3. Create a location file `<name>/<name>.py` following the standard pattern with two required functions:
   - `make_calib_pars()`: Returns a dictionary of calibration parameters specific to this location
   - `dataloader(location='<name>')`: Returns `fpld.DataLoader(location=location)`
   
   Reference existing locations like `locations/cotedivoire/cotedivoire.py` or `locations/senegal/senegal.py` for the exact pattern.
4. Add `from . import <name>` to `locations/__init__.py`.
5. At the top of `defaults.py` in the Global Defaults section, add the (lowercase) location name to the `valid_country_locs` array.

### To add regional model(s):
1. Create a new folder with the (lowercase) country name `<country>`, and a subdirectory called `regions`.
2. Create a folder `<country>/regions/data`; this is where any available region-specific data will be stored for use in the model. (See the Model Parameters table below for a list of all required files for the FPsim model to run.) The model depends on having all listed data files, but **if regional data for a given parameter is not available, you may substitute country-level data instead.** In such cases, be sure to include this country data in `locations/<country>/data`. Many data files have accompanying R scripts for preprocessing, while others—especially those derived from specific studies—may require manual formatting. Each region-level file should be stored in `locations/<country>/regions/data` and follow the schema found in `locations/kenya/data` **with the addition of a `region` column to regionally disaggregate the data** (see `locations/ethiopia/regions/data` for examples). Both country- and region-level filenames must match the naming conventions specified in the table(s) below.
3. For each region, create a region file `<country>/regions/<region>.py` following the standard pattern with:
   - `make_calib_pars()`: Returns region-specific calibration parameters
   - `dataloader(location='<region>')`: Returns `fpld.DataLoader(location=location)`
   
   Reference `locations/ethiopia/regions/addis_ababa.py` for the exact pattern.
4. Add `from .{country}.regions import {region}` to `locations/__init__.py`.
5. At the top of `defaults.py` in the Global Defaults section, add the country name as a key to the `valid_region_locs` dictionary with the region names listed in an array as its corresponding value (see `defaults.py` ethiopia regions as an example).

> Note: The experiment and calibration class have not been updated to run with regional models as of yet; however, the plotting class can be used to plot model output vs available data. 

## Running Pre-calibrated Simulations

To run a simulation with pre-calibrated parameters for any location, use the generic run script:

```bash
# Basic usage
python fpsim/locations/run_calibrated_location.py <location_name>

# Examples
python fpsim/locations/run_calibrated_location.py cotedivoire
python fpsim/locations/run_calibrated_location.py nigeria_lagos

# With custom parameters
python fpsim/locations/run_calibrated_location.py senegal --n-agents 1000 --end-year 2015

# Load existing results instead of running
python fpsim/locations/run_calibrated_location.py niger --load

# Quick run without saving or generating plots
python fpsim/locations/run_calibrated_location.py kenya --no-save --no-plots
```

This script works with any valid FPsim location and provides command-line options for customization.

## Adding a Location in an (external) Analysis Repo

To add a new location in an analysis repo:
1. Create a `locations` directory in your analysis repo, and copy into it the following from `fpsim/locations`:
   - `data_utils.py`
2. Create a folder `locations/<name>` with a subfolder `locations/<name>/data` and add your model data files there (see the 'Model Parameters' section below for the specific files required). Many of the data files have corresponding R processing scripts to format the data, and some are from specific studies and need to be formatted manually. For the proper format of each file, refer to the files in `locations/kenya/data`. **Note that the filenames MUST match the naming conventions defined in the table(s) below.**
3. Create `locations/<name>/<name>.py` following the standard pattern with `make_calib_pars()` and `dataloader()` functions (reference existing locations for the exact pattern).
4. In your analysis repository, you can now use your custom location by adding the following to the top of a script:
   - `from locations import <name>`
   - `fp.defaults.register_location('<name>', <name>)`
5. To run fpsim with your location, ensure you also have fpsim installed (either via pip or locally) so that the core model code can be used.

## Model Data
### Model Parameters

The following table lists the metrics used to parameterize any model within FPsim. Most data sources are context-specific 
and should be stored in the `locations/<name>/data/` directory. However, some data files are shared across locations (indicated 
by the `shared_data/` prefix in the Filename and location column), as they are derived from studies applicable to multiple settings.

Many of these data files are generated using the scripts listed in the 'Filename of processing code' column. These processing 
scripts are located in the `fpsim/data_processing/` directory. See the README in that directory for further guidance on 
generating and updating these datasets.


| Metric | Parameter or function name | Source | Filename and location | Filename of processing code |
|:---|:---|:---|:---|:----------------------------|
| Abortion probability | abortion_prob | [Source](https://bmcpregnancychildbirth.biomedcentral.com/articles/10.1186/s12884-015-0621-1) | scalar_probs.csv | N/A, See 'Source' column    |
| Twin probability | twins_prob | [Source](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0025239) | scalar_probs.csv | N/A, See 'Source' column         |
| Breastfeeding distribution mu value | breastfeeding_dur_mu | Requires children's recode DHS file | bf_stats.csv | breastfeeding_stats.R       |
| Breastfeeding distribution beta value | breastfeeding_dur_beta | Requires children's recode DHS file | bf_stats.csv | breastfeeding_stats.R       |
| Population pyramid | age_pyramid | UN World Population Prospects | age_pyramid.csv | UN_data_scraping.py         |
| Age-specific general mortality annual probabilities | age_mortality | UN Data Portal | mortality_prob.csv | UN_data_scraping.py         |
| Crude death rate trend | age_mortality | UN Data Portal | mortality_trend.csv | UN_data_scraping.py         |
| Urban proportion | urban_prop | DHS | urban.csv | extract_urban.py            |
| Maternal mortality | maternal_mortality | World Bank | maternal_mortality.csv | WorldBank_data_scraping.py  |
| Infant mortality | infant_mortality | World Bank, Noori et al (age gradient) | infant_mortality.csv, shared_data/age_adjustments.yaml | WorldBank_data_scraping.py  |
| Miscarriage rates | miscarriage_rates | [Magnus et al. 2019](https://pubmed.ncbi.nlm.nih.gov/30894356/) | shared_data/miscarriage.csv | N/A, See 'Source' column         |
| Stillbirth rates | stillbirth_rate | UN IGME, Noori et al. (age gradient) | stillbirths.csv, shared_data/age_adjustments.yaml |  N/A, See 'Source' column                           |
| Age-specific fecundity | age_fecundity | PRESTO study | shared_data/age_fecundity.csv |  N/A, See 'Source' column                           |
| Fecundity Ratio Nullip | fecundity_ratio_nullip | PRESTO study | shared_data/fecundity_ratio_nullip.csv |  N/A, See 'Source' column                           |
| LAM likelihood | lactational_amenorrhea | DHS | lam.csv | postpartum_recode.R         |
| Sexual activity last month (of debuted women) | sexual_activity | DHS | sexually_active.csv | sexual_activity.R           |
| Sexual activity postpartum | sexual_activity_pp | DHS | sexually_active_pp.csv | postpartum_recode.R         |
| Sexual debut | debut_age | DHS | debut_age.csv | sexual_debut_age_probs.py   |
| Contraceptive prevalence rate | mcpr | UN Data Portal | cpr.csv | UN_data_scraping.py         |
| Age-based Probability of Partnership | age_partnership | DHS | age_partnership.csv | age_partnership.R           |
| Wealth Quintile | wealth_quintile | DHS | wealth.csv | wealth.R                    |
| Education Objective | Education Objective | DHS | edu_objective.csv | education_dhs_pma.R         |
| Education Stop | Education Stop | PMA | edu_stop.csv | education_dhs_pma.R         |
| Education Initialization | Education Initialization | DHS | edu_initialization.csv | education_dhs_pma.R         |
| StandardChoice contraception function coefficients | Used in contraception functions | DHS | contra_coef_mid.csv, contra_coef_mid_pp1.csv | contra_coeffs.R             |
| SimpleChoice contraception function coefficients | Used in contraception functions | DHS | contra_coef_simple.csv, contra_coef_simple_pp1.csv | contra_coeffs.R             |
| Method mix matrices by age group, previous method, and postpartum status | Used to select contraceptive method | DHS | method_mix_matrix_switch.csv | make_matrices.R             |
| Time on method coefficients | Used in time on method distribution parameterization | DHS | method_time_coefficients.csv | time_on_method.R            |


### Calibration Targets
The data below are used to calibrate each FPsim model. During calibration, the model-generated outputs for these metrics 
are compared against real-world data to assess model accuracy and performance.

The corresponding processing scripts are located in the `fpsim/data_processing/` directory. For further guidance on the 
calibration workflow, see the examples in `fpsim/examples/` and `fpsim/examples/calibration_scripts/`.

| Calibration Target | Source         | Filename and location | Filename of processing code |
|:---|:---------------|:---|:----------------------------|
| Contraceptive prevalence rate | UN Data Portal | cpr.csv | UN_data_scraping.py         |
| Population size | World Bank     | popsize.csv | WorldBank_data_scraping.py  |
| Total fertility rate | World Bank     | tfr.csv | WorldBank_data_scraping.py  |
| Age-specific fertility rate | UN Data Portal | asfr.csv | UN_data_scraping.py         |
| MMR, IMR, CDR, CBR point targets | World Bank     | basic_wb.yaml | WorldBank_data_scraping.py  |
| Age-parity distribution  | DHS            | ageparity.csv | ageparity.R                 |
| Age at first birth | DHS            | afb.table.csv | age_at_first_birth.R        |
| Birth spacing | DHS            | birth_spacing_dhs.csv | birth_spacing.R             |
| Contraceptive method mix | DHS            | mix.csv | method_mix.R                |
| Contraceptive method use (binary) | DHS            | use.csv | method_mix.R                |


### Parameters (Assumptions for Calibration and Data Insights)
The parameters listed below represent model assumptions or free parameters that are not directly tied to empirical datasets. 
These values can be manually adjusted by the user to fine-tune model behavior, explore different scenarios, or support calibration 
when empirical data are limited or uncertain.

They are typically defined within the `<name>.py` model file or in `parameters.py` and may be changed directly 
by modifying the model configuration or parameter dictionary (pars). These parameters are especially useful for sensitivity 
analyses, policy experiments, or to reflect plausible variation across contexts.


| Parameter | Metric                                | Source | Filename and location                                                                                                                |
|:---|:--------------------------------------|:---|:-------------------------------------------------------------------------------------------------------------------------------------|
| Age-based conception exposure | exposure_age               | Calibration parameter | Manually input, model file                                                                                                           |
| Parity-based conception exposure | exposure_parity            | Calibration parameter | Manually input, model file                                                                                                           |
| Birth spacing preference | spacing_pref                          | Calibration parameter | birth_spacing_pref.csv; if this file is not created with assumed weights, the model will use weights of `1` for all month intervals. |
| Personal fecundity variation range | fecundity_low, fecundity_high | Calibration parameter | parameters.py, can be manually changed via pars[]                                                                                    |
| Overall exposure correction factor | exposure_factor                       | Calibration parameter | parameters.py, can be manually changed via pars[]                                                                                    |
| Primary infertility | primary_infertility                   | Calibration parameter | parameters.py, can be manually changed via pars[]                                                                                    |
