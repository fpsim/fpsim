'''
File to generate postpartum-specific probability curves informing parameters for
sexual activity and lactational amenorrhea (LAM)
Corresponds to Figure 3 of FPsim methods manuscript
- Sexual activity is calibrated to Senegal from DHS data asking about sexual activity in the last 4 weeks of
only postpartum women by postpartum month.
- LAM is calibrated to Senegal from DHS for breastfeeding women by month who are exclusively breastfeeding (bm + water)
and menses have not returned.  Limit of LAM use is 6 months.
Visualizations are from data in senegal_parameters.py
'''

import matplotlib.pyplot as plt
import numpy as np
import sciris as sc
from fp_analyses.methods_manuscript import base
from fp_analyses import senegal_parameters as sp

# Housekeeping
sc.tic()

plot_postpartum_curves = 1
do_save = 0

# Set up parameters
pars = sp.make_pars()

max_postpartum_months = 35

if plot_postpartum_curves:
    months = np.arange(max_postpartum_months + 1)

    fig, axs = plt.subplots()
    axs.plot(months, pars['sexual_activity_postpartum']['percent_active'], color='g', linewidth=2)
    axs.set_title('Sexual activity postpartum', fontsize=37)
    axs.set_ylabel('Probability per 1 month', fontsize=28)
    #axs[1].plot(months, pars['lactational_amenorrhea'], color='g', linewidth=2) #TODO- Need to build out LAM array to 35 months, add 24 more 0s
    #axs[1].set_title('Lactational amenorrhea', fontsize=37)
    #axs[1].set_ylabel('Probability per 1 month', fontsize=28)
    #axs[1].axvline(x=5, color='g', linestyle='--', label='limit of LAM use')
    #axs[1].legend(prop={"size": 18}, fancybox=True, framealpha=1, shadow=True, borderpad=1)
    axs.set_xlabel('Month postpartum', fontsize=40, fontweight='bold')

    #for ax in axs:
        #ax.tick_params(labelsize=30)

    fig.set_figwidth(12)
    fig.set_figheight(18)

    plt.show()

    if do_save:
        plt.savefig(base.abspath('output_files/postpartum_curves.png'))

sc.toc()