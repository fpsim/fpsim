'''
Explore dynamics around agents on methods with limited or no historical sexual activity
'''

import sciris as sc
import fpsim as fp
import pandas as pd
import seaborn as sns
import pylab as pl

# Set globals
max_age = 50
min_age = 15

# Set options
do_plot_sim = False

# Set up sim parameters
pars = fp.pars(location='kenya')
pars['n_agents'] = 100_000 # Small population size

sc.tic()
sim = fp.Sim(pars=pars)
sim.run()

if do_plot_sim:
    sim.plot()

# Save results
res = sim.results

# Save people from sim
ppl = sim.people

# Extract synthetic cohort of live women age 15-49 and find how many users haven't debuted

all_women = 0
method_users = 0
use_before_debut = 0

predebut = 0
sexually_infrequent = 0
adult_sexually_infrequent = 0
youth_sexually_infrequent = 0
adult_postdebut = 0
youth_postdebut = 0

time_to_debut = []

for i in range(len(ppl)):
    if ppl.alive[i] and not ppl.sex[i] and ppl.age[i] >= min_age and ppl.age[i] < max_age:
        all_women += 1
        if ppl.method[i] != 0:
            method_users += 1
        if ppl.sexual_debut[i] == 0 and ppl.method[i] != 0:
            use_before_debut += 1
        if ppl.sexual_debut[i] == 0:
            predebut += 1
        if ppl.age[i] >= ppl.fated_debut[i]:
            if ppl.sexual_debut[i] == 0:
                time_to_debut.append(ppl.age[i] - ppl.fated_debut[i])
            else:
                time_to_debut.append(ppl.sexual_debut_age[i] - ppl.fated_debut[i])
        if ppl.months_inactive[i] >= 12 and ppl.sexual_debut[i] == 1:
            sexually_infrequent += 1
        if ppl.months_inactive[i] >= 12 and ppl.sexual_debut[i] == 1 and ppl.age[i] >= 18:
            adult_sexually_infrequent += 1
        if ppl.months_inactive[i] >=12 and ppl.sexual_debut[i] == 1 and ppl.age[i] < 18:
            youth_sexually_infrequent += 1
        if ppl.sexual_debut[i] == 1 and ppl.age[i] >= 18:
            adult_postdebut += 1
        if ppl.sexual_debut[i] == 1 and ppl.age[i] < 18:
            youth_postdebut += 1

postdebut = all_women - predebut

ttd = pl.asarray(time_to_debut)

print(f'mean years from fated debut to first sex: {ttd.mean()}')

method_perc = method_users / all_women
use_before_debut_perc = use_before_debut / all_women

data_values = {'Method users': method_users,
                'Women 15-49': all_women,
                'Users before debut': use_before_debut
               }

data_percs =  {'CPR': method_users/all_women *100,
                'Percent users not debuted': use_before_debut/method_users *100,
                'CPR from non-debuters': use_before_debut/all_women *100
               }

data_sexually_infrequent = {'% 15-49 not yet': predebut/all_women *100,
                            '% inact 12+ mo, deb': sexually_infrequent/postdebut * 100,
                            '% inact 12+ mo, deb > 18': adult_sexually_infrequent/adult_postdebut * 100
                            }


per_frame = pd.DataFrame(data=data_percs, index=[0])
infrequent_frame = pd.DataFrame (data=data_sexually_infrequent, index=[0])
frame = pd.DataFrame({'Use':['All method users', 'Users before debut'], 'perc':[method_perc, use_before_debut_perc]})

print(data_values)
print(per_frame)
print (infrequent_frame)
print (f'% inact 12+ mo, deb < 18: {youth_sexually_infrequent/youth_postdebut * 100}')

#ax = frame.plot.bar(x='Use', y='perc', rot=90)
#ax.set_title('Method use before sexual debut - Senegal')

#pl.show()

sc.toc()