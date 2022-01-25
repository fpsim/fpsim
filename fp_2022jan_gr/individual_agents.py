'''
Records the individual states of a random agent who meets specific criteria
Generates a csv of agent states at time points throughout the sim
Used for plotting the life course of a single agent
'''


import sciris as sc
import fpsim as fp
import fp_analyses as fa
import pandas as pd
import random


do_save = 0

# A function to record the sim states over a run

data = dict()

def record(sim):

    ppl = sim.people
    entry = sc.objdict()
    entry.i = sim.i
    entry.y = sim.y
    entry.sex = sc.dcp(ppl.sex[:n])
    entry.age = sc.dcp(ppl.age[:n])
    entry.dead = sc.dcp(ppl.alive[:n])
    entry.active = sc.dcp(ppl.sexually_active[:n])
    entry.preg = sc.dcp(ppl.pregnant[:n])
    entry.method = sc.dcp(ppl.method[:n])
    entry.parity = sc.dcp(ppl.parity[:n])
    entry.lam = sc.dcp(ppl.lam[:n])
    entry.postpartum = sc.dcp(ppl.postpartum[:n])
    entry.fecundity = sc.dcp(ppl.personal_fecundity[:n])
    data[sim.i] = entry

    return

# Run the sim and record it
pars = fa.senegal_parameters.make_pars()
n = 5000
pars['n'] = n
pars['start_year'] = 1990
pars['interventions'] = [record]
sim = fp.Sim(pars)
sim.run()

# Pull out the people at the end of the sim
ppl = sim.people

# Set criteria for what kind of agent you'd like to track and then pick one at random
inds = sc.findinds((ppl.stillbirth == 1) * (ppl.alive == 1) * (ppl.parity >= 4) * (ppl.sex == 0) * (ppl.method != 0))
print(f'Indices meeting criteria: {inds}')
agent = random.choice(inds)
print(f'Index chosen of agent: {agent}')

# Go through entries and create dataframe of agent states of interest for manipulating
rows = []
with sc.timer('generating'):
    print('Generating...')
    for entry in list(data.values()):
        print(f'  Working on {entry.i} of {len(data)}...')
        row = {'i': None, 'age': None, 'active': None, 'method': None, 'pregnant': None, 'parity': None, 'postpartum': None,
               'personal fecundity': None, 'lam': None, 'age-based fecundity': None}
        row['i'] = entry.i
        row['age'] = entry.age[agent]
        age = int(entry.age[agent])
        row['active'] = entry.active[agent]
        row['method'] = entry.method[agent]
        row['pregnant'] = entry.preg[agent]
        row['parity'] = entry.parity[agent]
        row['postpartum'] = entry.postpartum[agent]
        row['lam'] = entry.lam[agent]
        row['personal fecundity'] = entry.fecundity[agent]
        row['age-based fecundity'] = (entry.fecundity[agent]) * (pars['age_fecundity'][age])

        rows.append(row)

# Save deliveries and stillbirths separately to be able to overlay as needed
rows_deliveries = []
for delivery in ppl.dobs[agent]:
    row = {'live deliveries': None}
    row['live deliveries'] = delivery
    rows_deliveries.append(row)

rows_stillbirths = []
for stillbirth in ppl.still_dates[agent]:
    row = {'stillbirths': None}
    row['stillbirths'] = stillbirth
    rows_stillbirths.append(row)

states = pd.DataFrame(data=rows)
deliveries = pd.DataFrame(data=rows_deliveries)
stillbirths = pd.DataFrame(data=rows_stillbirths)

if do_save:
    states.to_csv('/Users/Annie/model_postprocess_files/states_agent_'+{ppl.uid[agent]}+'.csv')
    deliveries.to_csv('/Users/Annie/model_postprocess_files/deliveries.csv')
    stillbirths.to_csv('/Users/Annie/model_postprocess_files/stillbirths.csv')


