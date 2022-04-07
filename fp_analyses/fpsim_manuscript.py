import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sciris as sc
import seaborn as sns

import fpsim as fp
from fp_analyses import senegal_parameters as sp

# Housekeeping
sc.tic()

do_plot_figure_2 = 1

# Set up sim
pars = sp.make_pars()
pars['n'] = 2000 # Adjust running parameters easily here
pars['start_year'] = 1960
sim = fp.Sim(pars)
sim.run()

# global variables for use in plotting
max_age = 50
max_postpartum_months = 35

# Plot age-specific FPsim parameters
if do_plot_figure_2:
    ages = np.arange(max_age + 1)

    fig1, axs = plt.subplots(3)
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


    for ax in axs:
        ax.tick_params(labelsize= 30)

    fig1.text(0.5, 0.001,'Age' ,fontsize=40, fontweight='bold', ha='center', va='center' )

    fig1.set_figwidth(20)
    fig1.set_figheight(38)

    plt.show()

    #plt.savefig('Age parameters.png', dpi = 600)


    #axs[3].plot(ages, nullip_interp_model(ages),  color='b', linewidth=2)
    #axs[3].set_title('Nulliparous fecundability correction', fontsize = 40)
    #axs[3].set_ylabel('Correction ratio nullip vs parous', fontsize = 30)
