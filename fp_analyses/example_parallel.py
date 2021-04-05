'''
Illustration of parallel run
'''

import sciris as sc
import fpsim as fp
import senegal_parameters as sp

def run(seed=1, n=200):
    pars = sp.make_pars()
    pars['seed'] = seed
    pars['n'] = n
    sim = fp.Sim(pars)
    sim.run()
    return sim


if __name__ == '__main__':

    T = sc.tic()
    sims = sc.parallelize(run, range(5))
    all_ppl = sc.mergedicts(*[s.people for s in sims])
    msim = sc.dcp(sims[0])
    msim.people = all_ppl
    for k in msim.results.keys():
        for s,sim in enumerate(sims):
            if s>0:
                msim.results[k] += sim.results[k]

    msim.plot()
    sc.toc(T)
    print('Done.')