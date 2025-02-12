#----------------------------------------------------------The baseline fitting-------------------------------------------------------------------------------------------------------------------------#
import sciris as sc
import fpsim as fp
import matplotlib.pyplot as plt
import numpy as np

def run_baseline():
    pars = dict(
        n_agents   = 46750,
        location   = 'nuhdss',
        start_year = 2012,
        end_year   = 2040,
    )

    sim = fp.Sim(pars)
    sim.run()
    sim.plot()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#  
#------------------------------------------------------------------------Gradual roll-out of campaigns-------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---------------------------------Running campaigns for each individual methods of family planning e.g IUD at a single coverage------------------------------------------------------------------------#

def run_impl_campaign():
    n_agents = 10_000
    start_year = 2024
    end_year = 2040
  # Define logistic growth parameters
    C_max = 0.9                                                                             # Maximum contraceptive uptake (90%)
    C_0 = 0.02                                                                              # Initial contraceptive uptake (5%)
    t0 = 2030                                                                               # Year when 50% of max coverage is reached
    r = 0.2                                                                                 # Growth rate (adjustable)
    a = (C_max / C_0) - 1                                                                   # Derived from initial coverage
    # Logistic growth function for campaign roll-out
    def logistic_campaign(year):
        return C_max / (1 + a * np.exp(-r * (year - t0)))
    # Generate a range of years and compute coverage

    gradual_scen = None
    for year in range(start_year, end_year + 1):
        coverage_effect = logistic_campaign(year)

        # Define dynamic scenario with logistic campaign roll-out
        gradual_scen += fp.make_scen(method='Implants', init_factor=coverage_effect, year=year)

    # Set up simulation parameters
    pars = fp.pars(location='nuhdss', n_agents=n_agents, start_year=start_year, end_year=end_year)
    # Define scenarios
    scens = fp.Scenarios(pars=pars, repeats=3)
    scens.add_scen(label='Baseline')
    scens.add_scen(gradual_scen, label='Gradual Campaign Roll-out')
    scens.run()
    scens.plot()




if __name__ == '__main__':
    #run_baseline()
    run_impl_campaign()
    #run_inj_campaign()
    #run_campaign_coverage_inj()
    #run_all_methods_campaign()
    #run_campaign_coverage()
    #run_male_inv()
    


