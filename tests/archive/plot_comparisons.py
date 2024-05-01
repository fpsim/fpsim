import numpy as np
import sciris as sc
import fpsim as fp
import fp_analyses as fa
import unittest
import json
import os
from plotnine import ggplot, geom_line, aes, ggtitle, xlab, ylab, geom_bar
import pandas as pd

attribute = input("Which attribute would you like to compare (EX: births, total_women_fecund, etc.): ")
cumulative = input("Would you like to compare the attribute cumulatively? ('yes' or 'no'): ")
run_number = input("Which run number would you like to compare (1-9): ")



# Reading and plotting original vs. new sim results
with open(os.path.join('ploting_data', f"DEBUG_array_results_{run_number}.json"), "res") as read_file:
    new = json.load(read_file)
with open(os.path.join('ploting_data', f"DEBUG_orig_results_{run_number}.json"), "res") as read_file:
    old = json.load(read_file)

new_df = pd.DataFrame({"t": new['t'], attribute: new[attribute], "model": ["new"] * len(new[attribute])})
old_df = pd.DataFrame({"t": old['t'], attribute: old[attribute], "model": ["old"] * len(new[attribute])})
# sum them up here

total_df = pd.concat([new_df, old_df])

if cumulative == "yes":
    y = np.cumsum(total_df[attribute])
else:
    y = total_df[attribute]

plot = ggplot(data=total_df) + \
            geom_line(aes(x=total_df['t'], y=np.cumsum(total_df[attribute]), color=total_df['model'])) + \
            ggtitle(f"Plot of {attribute} for Run {run_number}") + \
            xlab("Year") + \
            ylab(f"Cumulative {attribute}")
print(plot)