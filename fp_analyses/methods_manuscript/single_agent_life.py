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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl


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
n = 50000
pars['n'] = n
pars['start_year'] = 1990
pars['interventions'] = [record]
sim = fp.Sim(pars)
sim.run()

# Pull out the people at the end of the sim
ppl = sim.people

# Set criteria for what kind of agent you'd like to track and then pick one at random
inds = sc.findinds((ppl.stillbirth != 0) * (ppl.alive == 1) * (ppl.parity >= 3) * (ppl.sex == 0) * (ppl.method != 0))
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
               'personal fecundability': None, 'lam': None, 'age-based fecundability': None}
        row['i'] = entry.i
        row['age'] = entry.age[agent]
        age = int(entry.age[agent])
        row['active'] = entry.active[agent]
        row['method'] = entry.method[agent]
        row['pregnant'] = entry.preg[agent]
        row['parity'] = entry.parity[agent]
        row['postpartum'] = entry.postpartum[agent]
        row['lam'] = entry.lam[agent]
        row['personal fecundability'] = entry.fecundity[agent]
        row['age-based fecundability'] = (entry.fecundity[agent]) * (pars['age_fecundity'][age])

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

rows_miscarriages = []
for miscarriage in ppl.miscarriage_dates[agent]:
    row = {'miscarriages': None}
    row['miscarriages'] = miscarriage
    rows_miscarriages.append(row)

rows_abortions = []
for abortion in ppl.abortion_dates[agent]:
    row = {'abortions': None}
    row['abortions'] = abortion
    rows_abortions.append(row)

states = pd.DataFrame(data=rows)
deliveries = pd.DataFrame(data=rows_deliveries)
stillbirths = pd.DataFrame(data=rows_stillbirths)
miscarriages = pd.DataFrame(data=rows_miscarriages)
abortions = pd.DataFrame(data=rows_abortions)

if do_save:
    states.to_csv('/Users/Annie/model_postprocess_files/states_agent_'+str(ppl.uid[agent])+'.csv')
    deliveries.to_csv('/Users/Annie/model_postprocess_files/deliveries_'+str(ppl.uid[agent])+'.csv')
    stillbirths.to_csv('/Users/Annie/model_postprocess_files/stillbirths_'+str(ppl.uid[agent])+'.csv')
    miscarriages.to_csv('/Users/Annie/model_postprocess_files/miscarriages_' +str(ppl.uid[agent])+ '.csv')
    abortions.to_csv('/Users/Annie/model_postprocess_files/abortions_' +str(ppl.uid[agent])+ '.csv')

# Make visualization of states of an agent based on criteria and sim run
# Process dataframe
states = states.replace(False, np.nan)
states = states.replace(True, 1)
states['on method'] = np.where((states['method'] != 0), 1, np.nan)

fig, ax = plt.subplots(figsize=(40, 15))

n = 10  # Number of colors needed from palette
cmap = plt.get_cmap('tab20', n)

colors = []

for i in range(cmap.N):
    rgba = cmap(i)
    colors.append(cl.rgb2hex(rgba))

plot_1 = ax.plot(states['age'], states['active'], color='#83EEFF', alpha=0.8, linewidth=30, label='Sexually active')
plot_2 = ax.plot(states['age'], states['lam'], color='#228B22', linewidth=22, label='LAM')
plot_3 = ax.plot(states['age'], states['postpartum'], color=colors[7], alpha=0.5, linewidth=16, label='Postpartum')
plot_4 = ax.plot(states['age'], states['pregnant'], color='#FF198C', linewidth=16, label='Pregnant')
plot_5 = ax.plot(states['age'], states['on method'], color='#ff4500', alpha=0.4, linewidth=45, label='On method')

if not deliveries.empty:
    for x in deliveries['live deliveries'][:-1]:
        ax.axvline(x=x, color='#000000', linestyle='--', linewidth=4)
    ax.axvline(x=deliveries['live deliveries'].iloc[-1], color='#000000', linestyle='--', linewidth=4,
           label="Live delivery")

if not stillbirths.empty:
    for s in stillbirths['stillbirths'][:-1]:
        ax.axvline(x=s, color=colors[3], linestyle='--', linewidth=4)
    ax.axvline(x=stillbirths['stillbirths'].iloc[-1], color=colors[3], linestyle='--', linewidth=4, label="Stillbirth")

if not miscarriages.empty:
    for m in miscarriages['miscarriages'][:-1]:
        ax.axvline(x=m, color=colors[2], linestyle='--', linewidth=4)
    ax.axvline(x=miscarriages['miscarriages'].iloc[-1], color=colors[2], linestyle='--', linewidth=4,
               label="Miscarriage")

if not abortions.empty:
    for a in abortions['abortions'][:-1]:
        ax.axvline(x=a, color=colors[1], linestyle= '--', linewidth=4)
    ax.axvline(x = abortions['abortions'].iloc[-1], color = colors[1], linestyle = '--', linewidth = 4, label = "Abortion")


ax.legend(prop={"size": 35}, fancybox=True, framealpha=1, shadow=True, borderpad=1, bbox_to_anchor=(1.25, 0.75))

ax.set_yticks([])

ax2 = ax.twinx()
plot_6 = ax2.plot(states['age'], states['age-based fecundability'], color=colors[0], marker=",", linewidth=4,
                  label="Age-based\n fecundability")

# Need to print out df for agent states to look at which methods they are using along the timeline and then manually add the method name at the right location
# x corresponds to agents age when they take up method, y needs to be adjusted for individual's plot

#ax2.annotate('Injectables', (25.8, 0.35), fontsize=30, fontweight='bold')
#ax2.annotate('Condoms', (28.9, 0.35), fontsize=30, fontweight='bold')
#ax2.annotate('Implants', (32, 0.35), fontsize=30, fontweight='bold')
#ax2.annotate('Pills', (35, 0.35), fontsize=30, fontweight='bold')


ax2.set_ylabel("Conceptions per woman per year of activity", fontsize=35)
ax.set_xlabel("Age", fontsize=35, fontweight='bold')

ax2.legend(prop={"size": 35}, fancybox=True, framealpha=1, shadow=True, borderpad=1)

ax.tick_params(labelsize=40)
ax2.tick_params(labelsize=40)

filename = "Life of an agent " + str(agent) + ".png"

plt.show()
#plt.savefig('filename')