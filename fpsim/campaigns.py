import fpsim as fp
import numpy as np
#--------------------------------------------------------Static and continous campaign implementation--------------------------------------------------------------------------------------------------#"
n_agents = 10_000
start_year = 2000
end_year = 2040

effect_size = 0.6
coverage = 0.75
init_factor = 1.0 + effect_size * coverage

scen = fp.make_scen(method='Implants', init_factor=init_factor, year=2025)

pars = fp.pars(location='nuhdss', n_agents=n_agents, start_year=start_year, end_year=end_year)

scens = fp.Scenarios(pars=pars, repeats=3)
scens.add_scen(label='Baseline')
scens.add_scen(scen, label='Campaign')
scens.run()
scens.plot()

#"-------------------------------------------------implementation of the logistic growth model for the campaigns---------------------------------------------------------------------------------------"#

def logistic_growth(t, C_max, C0, r, t0):
    a = (C_max / C0) - 1
    return C_max / (1 + a * np.exp(-r * (t - t0)))

# Parameters
C_max = 1.0  # Maximum achievable coverage (100%)
C0 = 0.1     # Initial campaign coverage (10%)
r = 0.5      # Growth rate
t0 = 10      # Time at which 50% of C_max is achieved


# Compute campaign coverage over time
coverage_logis = logistic_growth(time, C_max, C0, r, t0)


