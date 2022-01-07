import fpsim as fp
import senegal_parameters as sp
import pandas as pd

run_key = 'baseline'

do_run = 1
do_save = 1

if do_run:
    pars = sp.make_pars()
    sim = fp.Sim(pars=pars)
    sim.run()

    res = sim.results
    ppl = sim.people

    data_all_steps = {'time': res['t'], 'infant deaths': res['infant_deaths'],
            'total births': res['total_births'], 'mcpr': res['mcpr']}

    data_annual = {'time': res['tfr_years'],
                   'mcpr by year': res['mcpr_by_year'],
                   'infant deaths over year': res['infant_deaths_over_year'],
                   'total births over year': res['total_births_over_year']}

    df_all = pd.DataFrame(data_all_steps)
    df_annual = pd.DataFrame(data_annual)

    df_annual['im rate'] = (df_annual['infant deaths over year'] / df_annual['total births over year']) * 1000

if do_save:
   df_annual.to_csv('/Users/Annie/model_postprocess_files/infant_deaths_'+run_key+'.csv')  # Change filepath to yours

