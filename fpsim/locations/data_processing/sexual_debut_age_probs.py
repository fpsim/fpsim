import pandas as pd
import math


# v525 - age at first sexual intercourse
# v531 - age at first sex (imputed with age at first union)

data_file = 'KEIR72FL.DTA'

# Need raw.dta DHS data file here. Using Senegal DHS 2019.
dhs = pd.read_stata(data_file, convert_categoricals = False)  # Substitue your country's DHS data file here
dhs['wgt'] = dhs['v005']/1000000

dhs_debut = dhs[(dhs['v531'] != 0) & (dhs['v531'] < 50) & (dhs['v531'] >= 10)]
age_probs = dhs_debut[['v531', 'wgt']].groupby(['v531']).sum()
age_probs.reset_index()

age_probs['prob_weights'] = age_probs['wgt']/age_probs['wgt'].sum()
probs_sum = age_probs['prob_weights'].sum()
assert math.isclose(1, probs_sum), f'Sexual debut age probabilities should add to 1, not {probs_sum}'

age_probs['prob_weights'].to_csv('Debut_age_probs_DHS.csv')