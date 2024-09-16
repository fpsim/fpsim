import fpsim as fp

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