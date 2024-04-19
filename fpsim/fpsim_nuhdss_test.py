import fpsim as fp

pars = dict(
    n_agents   = 756,
    location   = 'kenya',
    start_year = 2012,
    end_year   = 2035,
)

sim = fp.Sim(pars)
sim.run()
fig = sim.plot()



#sim.to_df()
#sim.df.to_csv(r'results.csv')
#print("Done.")

