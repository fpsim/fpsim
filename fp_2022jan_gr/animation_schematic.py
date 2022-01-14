'''
Plot animation of agents in FPsim
'''

import os
import numpy as np
import sciris as sc
import pylab as pl
import fpsim as fp
import fp_analyses as fa


n_side = 10
n = n_side**2
filename = 'animation_data.obj'
rerun = False

if rerun or not os.path.exists(filename):

    data = dict()

    def record(sim):
        entry = sc.objdict()
        entry.i = sim.i
        entry.y = sim.y
        entry.method = sc.dcp(sim.people.method)
        entry.children = sc.dcp(sim.people.children)
        data[sim.i] = entry
        return

    pars = fa.senegal_parameters.make_pars()
    pars['n'] = n
    pars['start_year'] = 1990
    pars['interventions'] = [record]
    sim = fp.Sim(pars)
    sim.run()

    sim.plot()

    sc.save(filename, data)

else:

    data = sc.load(filename)


#%% Plot

max_children = 9
max_method = 10
radius = 0.2

x = np.arange(n_side)
y = np.arange(n_side)
xx,yy = np.meshgrid(x,y)
xx = xx.flatten()
yy = yy.flatten()

sc.options(dpi=200)
fig,ax = pl.subplots(figsize=(6,6))
sc.figlayout()

for entry in data.values():
    pl.cla()
    # cc = sc.vectocolor(entry.method[:n], maxval=max_method)
    cc = entry.method[:n] > 0
    pl.scatter(xx, yy, s=100, c=cc, marker='o')
    for m, (ix,iy) in enumerate(zip(xx,yy)):
        n_children = len(entry.children[m])
        if n_children:
            c_arr = np.arange(n_children)
            rad = 2*np.pi*c_arr/max_children
            dx = radius*np.cos(rad)
            dy = radius*np.sin(rad)
            pl.scatter(ix+dx, iy+dy, s=10, c='k')

    δ = 0.5
    pl.axis('off')
    pl.xlim((-δ, n_side-1+δ))
    pl.ylim((-δ, n_side-1+δ))
    pl.title(f'{entry.y:0.1f}')
    pl.pause(0.01)



print('Done.')
