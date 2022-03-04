import pandas as pd
import numpy as np
import sciris as sc
import fpsim as fp
import fp_analyses as fa

# Make sim
pars = fa.senegal_parameters.make_pars()
pars['n'] = 1000

exp = fp.ExperimentVerbose(pars)
exp.run(keep_people = True)
people = exp.sim.people

# Extrapolate date from people
months = {} # months: people

dobs_list = [np.diff(date_list) for date_list in people.dobs]

dobs = [round(item * 12) for sublist in dobs_list for item in sublist]
for spacing in dobs:
    if spacing not in months:
        months[spacing] = 0
    months[spacing] = months[spacing] + 1
for key in months:
    months[key] = months[key] / len(dobs)

# fill in the gaps
for i in range(max(months.keys())):
    if i not in months:
        months[i] = 0

# Split into sorted columns
month_bin = []
proportions = []
for key in sorted(months):
    month_bin.append(key)
    proportions.append(months[key])

dobs_csv = pd.DataFrame({"Month": month_bin, "Proportion": proportions})
dobs_csv.to_csv("DOB.csv", index=False)


