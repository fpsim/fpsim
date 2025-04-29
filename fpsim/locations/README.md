# Locations

This folder stores the location-specific data for FPsim. 

To add a new location:

1. Create a new folder with the (lowercase) location name `<name>`.
2. Create a folder `<name>/data` and add the source data files there.
3. Create a file `<name>.py` that is used to generate the location parameter values; see `senegal.py` for the correct structure.
4. Add `from . import <name>` to `__init__.py`.

### Model Parameters
The following are the metrics used to parameterize any model for FPsim. Most of the data for these metrics are context-specific 
and their corresponding data files should be stored in the `<name>/data` directory. There are a few data sources that are shared across locations (denoted with
the `shared_data/` prefix in the 'Filename and location' column), as they are derived from various studies applicable to multiple contexts. 
Many of these data files are created using processing code indicated in the 'Filename of processing code' column; each of 
these scripts can be found in the `fpsim/data_processing` directory. See also the README in this directory for further guidance.

| Metric | Parameter or function name | Source | Filename and location | Filename of processing code |
|:---|:---|:---|:---|:---|
| Abortion probability | abortion_prob | [Source](https://bmcpregnancychildbirth.biomedcentral.com/articles/10.1186/s12884-015-0621-1) | scalar_probs.csv | |
| Twin probability | twins_prob | [Source](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0025239) | scalar_probs.csv | |
| Breastfeeding distribution mu value | breastfeeding_dur_mu | Requires children's recode DHS file | bf_stats.csv | breastfeeding_stats.R |
| Breastfeeding distribution beta value | breastfeeding_dur_beta | Requires children's recode DHS file | bf_stats.csv | breastfeeding_stats.R |
| Population pyramid | age_pyramid | UN World Population Prospects | age_pyramid.csv | UN_data_scraping.py |
| Age-specific general mortality annual probabilities | age_mortality | UN Data Portal | mortality_prob.csv | UN_data_scraping.py |
| Crude death rate trend | age_mortality | UN Data Portal | mortality_trend.csv | UN_data_scraping.py |
| Urban proportion | urban_prop | DHS | urban.csv | extract_urban_from_dhs.py |
| Maternal mortality | maternal_mortality | World Bank | maternal_mortality.csv | WorldBank_data_scraping.py |
| Infant mortality | infant_mortality | World Bank, Noori et al (age gradient) | infant_mortality.csv, shared_data/age_adjustments.yaml | WorldBank_data_scraping.py |
| Miscarriage rates | miscarriage_rates | [Magnus et al. 2019](https://pubmed.ncbi.nlm.nih.gov/30894356/) | shared_data/miscarriage.csv | |
| Stillbirth rates | stillbirth_rate | UN IGME, Noori et al. (age gradient) | stillbirths.csv, shared_data/age_adjustments.yaml | |
| Age-specific fecundity | age_fecundity | PRESTO study | shared_data/age_fecundity.csv | |
| Fecundity Ratio Nullip | fecundity_ratio_nullip | PRESTO study | shared_data/fecundity_ratio_nullip.csv | |
| LAM likelihood | lactational_amenorrhea | DHS | lam.csv | postpartum_recode.R |
| Sexual activity last month (of debuted women) | sexual_activity | DHS | sexually_active.csv | sexual_activity.R |
| Sexual activity postpartum | sexual_activity_pp | DHS | sexually_active_pp.csv | postpartum_recode.R |
| Sexual debut | debut_age | DHS | debut_age.csv | sexual_debut_age_probs.py |
| Barriers to use | barriers | DHS | Manually input, model file | |
| Contraceptive prevalence rate | mcpr | UN Data Portal | cpr.csv | UN_data_scraping.py |
| Age-based Probability of Partnership | age_partnership | DHS | age_partnership.csv | age_partnership_dhs.R |
| Wealth Quintile | wealth_quintile | DHS | wealth.csv | wealth.R |
| Education Objective | Education Objective | DHS | edu_objective.csv | education_dhs_pma.R |
| Education Stop | Education Stop | PMA | edu_stop.csv | education_dhs_pma.R |
| Education Initialization | Education Initialization | DHS | edu_initialization.csv | education_dhs_pma.R |
| StandardChoice contraception function coefficients | Used in contraception functions | DHS | contra_coef_mid.csv, contra_coef_mid_pp1.csv | functions_DHS.R |
| SimpleChoice contraception function coefficients | Used in contraception functions | DHS | contra_coef_simple.csv, contra_coef_simple_pp1.csv | functions_DHS.R |
| Method mix matrices by age group, previous method, and postpartum status | Used to select contraceptive method | DHS | method_mix_matrix_switch.csv | make_matrices_dhs.R |
| Time on method coefficients | Used in time on method distribution parameterization | DHS | method_time_coefficients.csv | time_on_method.R and process_durations.py |


### Calibration Targets
These data are utilized for calibrating each FPsim model. The processing code files can be found in the `fpsim/data_processing` 
directory. In the calibration process, the calibration targets below generated from a model simulation will be compared 
to the country/context data for the same metrics to assess model accuracy. For further guidance on calibration, refer to 
the examples in the `fpsim/examples` and `fpsim/examples/calibration_scripts` directories.

| Calibration Target | Source         | Filename and location | Filename of processing code |
|:---|:---|:---|:---|
| Contraceptive prevalence rate | UN Data Portal | cpr.csv | UN_data_scraping.py |
| Population size | World Bank     | popsize.csv | WorldBank_data_scraping.py |
| Total fertility rate | World Bank     | tfr.csv | WorldBank_data_scraping.py |
| Age-specific fertility rate | UN Data Portal | asfr.csv | UN_data_scraping.py |
| MMR, IMR, CDR, CBR point targets | World Bank     | basic_wb.yaml | WorldBank_data_scraping.py |
| Age-parity distribution (skyscrapers) | DHS or PMA     | ageparity.csv | ageparity.R |
| Age at first birth | DHS            | afb.table.csv | age_at_first_birth_DHS.R |
| Birth spacing | DHS            | birth_spacing_dhs.csv | birth_spacing.R |
| Contraceptive method mix | PMA            | mix.csv | method_mix_PMA.R |
| Contraceptive method use (binary) | PMA            | use.csv | method_mix_PMA.R |


### Parameters (Assumptions for Calibration and Data Insights)

| Parameter | Metric | Source | Filename and location |
|:---|:---|:---|:---|
| Age-based conception exposure | exposure_age | Calibration parameter | Manually input, model file |
| Parity-based conception exposure | exposure_parity | Calibration parameter | Manually input, model file |
| Birth spacing preference | spacing_pref | Calibration parameter | birth_spacing_pref.csv |
| Personal fecundity variation range | fecundity_var_low, fecundity_var_high | Calibration parameter | parameters.py, can be manually changed via pars[] |
| Overall exposure correction factor | exposure_factor | Calibration parameter | parameters.py, can be manually changed via pars[] |
| Primary infertility | primary_infertility | Calibration parameter | parameters.py, can be manually changed via pars[] |
