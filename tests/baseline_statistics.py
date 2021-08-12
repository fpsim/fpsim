'''
Like test_baselines.py, but with statistics.
'''

import numpy as np
import pandas as pd
import sciris as sc
import fpsim as fp
import fp_analyses as fa

rerun   = 1  # Re-run rather than just load
do_save = 1  # Re-save, overwriting existing file
n_runs  = 20 # Number of runs to run
outfile = 'baseline_statistics.df'

def make_exp(n=500, seed=1, verbose=0.1, do_run=False, do_plot=False):
    '''
    Define a default simulation for testing the baseline.
    '''
    pars = fa.senegal_parameters.make_pars()
    pars['n'] = n
    pars['seed'] = seed
    pars['verbose'] = verbose
    exp = fp.Experiment(pars=pars)

    if do_run or do_plot:
        exp.run()

    if do_plot:
        exp.plot()

    return exp


def run_save_sims(n_runs=n_runs, do_save=do_save):

    x = np.arange(n_runs)
    exps = sc.parallelize(make_exp, iterkwargs=dict(seed=x+1), kwargs=dict(do_run=True))
    data = {}
    cols = []
    for i in x:
        col = f'seed{i+1}'
        cols.append(col)
        data[col] = exps[i].summarize()['model']

    rawdf = pd.DataFrame(data)
    df = pd.DataFrame()
    df['mean'] = rawdf[cols].mean(axis=1)
    df['std']  = rawdf[cols].std(axis=1)
    for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
        df[f'q{q}'] = rawdf[cols].quantile(q=q, axis=1)

    out = sc.objdict(df=df, rawdf=rawdf)
    if do_save:
        sc.saveobj(outfile, out)

    return out


if __name__ == '__main__':

    sc.tic()
    old = sc.loadobj(outfile)
    if rerun:
        new = run_save_sims(n_runs=n_runs, do_save=do_save)
    sc.toc()
