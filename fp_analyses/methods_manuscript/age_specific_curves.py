'''
File to generate age-specific probability curves informing parameters for
fecundability, sexual activity, and miscarriage
Corresponds to Figure 2 of FPsim methods manuscript
- Sexual activity is calibrated to Senegal from DHS data asking about sexual activity in the last 4 weeks.
- Fecundability is from the PRESTO study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5712257/
- Miscarriage probability is from Magnus et al BMJ 2019: https://pubmed.ncbi.nlm.nih.gov/30894356/
Visualizations are from data in senegal_parameters.py
'''

import matplotlib.pyplot as plt
import numpy as np
import sciris as sc
from fp_analyses.methods_manuscript import base
import fpsim as fp
from fp_analyses import senegal_parameters as sp

# Housekeeping
sc.tic()

plot_age_specific_curves = 1
do_save = 1

# Set up parameters
pars = sp.make_pars()
#pars['n'] = 2000 # Adjust running parameters easily here
#pars['start_year'] = 1960
#sim = fp.Sim(pars)
#sim.run()

# global variables for use in plotting
max_age = 50

# Plot age-specific FPsim parameters
if plot_age_specific_curves:
    ages = np.arange(max_age + 1)

    fig, axs = plt.subplots(3)
    axs[0].plot(ages, pars['age_fecundity'], color='b', linewidth=2)
    axs[0].set_title('Fecundability', fontsize = 40)
    axs[0].set_ylabel('Probability conception per 12 months', fontsize = 30)
    #axs[0].fill_between(ages, (fecundity_interp_model(ages)-fecundity_interp_model(ages)*fecundity_variation[0]),
                    #(fecundity_interp_model(ages)+fecundity_interp_model(ages)*fecundity_variation[1]), color='b', alpha=.1)
    axs[1].plot(ages, pars['sexual_activity'], color='b', linewidth=2)
    axs[1].set_title('Sexual activity', fontsize = 40)
    axs[1].set_ylabel('Probability per 1 month', fontsize = 30)
    axs[2].plot(ages, pars['miscarriage_rates'], color='b', linewidth=2)
    axs[2].set_title('Miscarriage', fontsize = 40)
    axs[2].set_ylabel('Probability per pregnancy', fontsize = 30)
    axs[2].set_xlabel('Age', fontweight='bold', fontsize=50)

    for ax in axs:
        ax.tick_params(labelsize= 30)

    fig.set_figwidth(20)
    fig.set_figheight(40)

    plt.show()

    if do_save:
        plt.savefig(base.abspath('output_files/age_parameter_curves.png'))

sc.toc()
