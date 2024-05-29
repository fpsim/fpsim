"""
Test running sims
"""

import fpsim as fp
import sciris as sc
import numpy as np
import pandas as pd

par_kwargs = dict(n_agents=100, start_year=2000, end_year=2010, seed=1, verbose=1)


def test_simple():
    sc.heading('Test simplest possible FPsim run')
    sim = fp.Sim(location='test')
    sim.run()
    sim.plot()
    return sim


def test_simple_choice(location='kenya'):
    sc.heading('Choose method annually based on age and parity')

    # Read in data
    coef_df = pd.read_csv(f'../fpsim/locations/{location}/data/contra_coef_simple.csv')
    coef = sc.objdict(
        intercept=coef_df.Estimate[0],
        age_bin_vals=coef_df.Estimate[1:].values,
        age_bin_edges=[18, 20, 25, 35, 50],
    )
    coef_pp1_df = pd.read_csv(f'../fpsim/locations/{location}/data/contra_coef_simple_pp1.csv')
    coef_pp1 = sc.objdict(
        intercept=coef_pp1_df.Estimate[0],
        age_bin_vals=coef_pp1_df.Estimate[1:].values,
        age_bin_edges=[18, 20, 25, 35],
    )
    coef_pp6_df = pd.read_csv(f'../fpsim/locations/{location}/data/contra_coef_simple_pp6.csv')
    coef_pp6 = sc.objdict(
        intercept=coef_pp6_df.Estimate[0],
        age_bin_vals=coef_pp6_df.Estimate[1:].values,
        age_bin_edges=[18, 20, 25, 35],
    )

    methods = fp.Methods

    # Read in duration estimates
    dur_raw = pd.read_csv(f'../fpsim/locations/{location}/data/method_time_coefficients.csv', keep_default_na=False, na_values=['NaN'])
    for method in methods.values():
        mlabel = method.csv_name
        if mlabel != 'F.sterilization':
            thisdf = dur_raw.loc[dur_raw.method==mlabel]
            dist = thisdf.functionform.iloc[0]
            if dist == 'lognormal':
                method.dur_use['dist'] = dist
                method.dur_use['par1'] = thisdf.coef[thisdf.estimate=='meanlog'].values[0]
                method.dur_use['par2'] = thisdf.coef[thisdf.estimate=='sdlog'].values[0]
                method.dur_use['age_bin_vals'] = thisdf.coef.values[2:]
                method.dur_use['age_bin_edges'] = [18, 20, 25, 35]

    method_choice = fp.SimpleChoice(methods=methods, coef=coef, coef_pp1=coef_pp1, coef_pp6=coef_pp6)
    pars = fp.pars(location=location, **par_kwargs)
    s = fp.Sim(pars, contraception_module=method_choice)
    s.run()
    return s


def test_empowered_choice():
    sc.heading('Test sim with empowerment')

    # Define new modules
    ms = fp.EmpoweredChoice(location='kenya')
    emp = fp.Empowerment(location='kenya')
    edu = fp.Education(location='kenya')

    # Define pars
    pars = fp.pars(location='kenya', **par_kwargs)

    # Make and run sim
    s = fp.Sim(pars, contraception_module=ms, empowerment_module=emp, education_module=edu)
    s.run()

    return s


if __name__ == '__main__':

    # s0 = test_simple()
    s1 = test_simple_choice()
    # s2 = test_empowered_choice()
