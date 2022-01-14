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
rerun  = 0
doplot = 1
dosave = 1

if rerun or not os.path.exists(filename):

    data = dict()

    def record(sim):
        entry = sc.objdict()
        entry.i = sim.i
        entry.y = sim.y
        entry.dead     = sc.dcp(~sim.people.alive)
        entry.active   = sc.dcp(sim.people.sexually_active)
        entry.preg     = sc.dcp(sim.people.pregnant)
        entry.method   = sc.dcp(sim.people.method)
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

if doplot:

    sc.options(font='Avenir')

    cmap = sc.objdict(
        inactive = '#000',
        active   = '#0a0',
        preg     = '#ff0',
        method   = '#f00',
        dead     = '#ccc',
    )

    cnames = sc.objdict(
        inactive = 'Not sexually\nactive',
        active   = 'Sexually\nactive',
        preg     = 'Pregnant',
        method   = 'Using\ncontraception',
        dead     = 'Dead',
    )

    max_children = 9
    max_method = 10
    radius = 0.2
    mothersize = 100
    childsize = 10

    x = np.arange(n_side)
    y = np.arange(n_side)
    xx,yy = np.meshgrid(x,y)
    xx = xx.flatten()
    yy = yy.flatten()

    sc.options(dpi=200)
    fig = pl.figure(figsize=(7.5,6))
    ax = pl.axes([0,0,1,1])
    δ = 0.5
    pl.axis('off')
    # sc.figlayout(top=0.93, right=0.72)

    frames = []

    def xtr(x):
        return 0.05+x/(n_side+3.5)

    def ytr(y):
        return 0.05+y/(n_side+1)

    stride = 2 # Don't render every frame
    with sc.timer('generating'):
        print('Generating...')
        for entry in list(data.values())[::stride]:
            print(f'  Working on {entry.i} of {len(data)}...')
            frame = sc.autolist()
            if not dosave:
                ax.clear()

            # Handle counts
            counts = sc.objdict()
            cc = np.array([cmap.inactive]*n, dtype=object)
            colorkeys = ['active', 'preg', 'method', 'dead']
            for key in colorkeys:
                inds = sc.findinds(entry[key][:n])
                cc[inds] = cmap[key]
                counts[key] = len(inds)
            counts.inactive = n - counts[:].sum()
            alive = n - counts.dead
            percents = sc.objdict()
            percents.inactive = sc.safedivide(counts.inactive, alive) * 100
            percents.active   = sc.safedivide(counts.active, alive) * 100
            percents.preg     = sc.safedivide(counts.preg, alive) * 100
            percents.method   = sc.safedivide(counts.method, counts.active) * 100
            percents.dead     = sc.safedivide(counts.dead, n) * 100

            # Plot legend -- have to do manually since legend not supported by animation
            dy = 0.7
            for i,key,label in cnames.enumitems():
                kwargs = dict(horizontalalignment='left', verticalalignment='center')
                x = n_side + 0.2
                x2 = x + 0.25
                y = n_side - 2.5 - dy*i
                y2 = y - 0.05
                frame += ax.scatter(xtr(x), ytr(y), s=mothersize, c=cmap[key])
                frame += ax.text(xtr(x2), ytr(y2), f'{label} ({percents[key]:0.0f}%)', **kwargs)

            y3 = y2 + 4
            y4 = y3 - dy/2
            frame += ax.scatter(xtr(x), ytr(y3), s=mothersize, c='k')
            frame += ax.scatter(xtr(x), ytr(y4), s=childsize, c='k')
            frame += ax.text(xtr(x2), ytr(y3), 'Woman', **kwargs)
            frame += ax.text(xtr(x2), ytr(y4), 'Child', **kwargs)

            # Actually plot
            frame += pl.scatter(xtr(xx), ytr(yy), s=mothersize, c=cc)
            for m, (ix,iy) in enumerate(zip(xx,yy)):
                n_children = len(entry.children[m])
                if n_children:
                    c_arr = np.arange(n_children)
                    rad = 2*np.pi*c_arr/max_children - np.pi/2
                    dx = radius*np.cos(rad)
                    dy = radius*np.sin(rad)
                    frame += ax.scatter(xtr(ix+dx), ytr(iy+dy), s=childsize, c='k')

            kwargs = dict(horizontalalignment='center', fontweight='bold') # Set the "title" properties
            frame += ax.text(0.4, 0.93, f'Year {entry.y:0.1f}', **kwargs) # Unfortunately pl.title() can't be dynamically updated
            frames.append(frame)
            ax.set_xlim((0, 1))
            ax.set_ylim((0, 1))
            ax.axis('off')
            if not dosave:
                pl.pause(0.01)

    if dosave:
        with sc.timer('saving'):
            print('Saving...')
            sc.savemovie(frames, 'fp_animation_schematic.mp4', fps=10, quality='high') # Save movie as a high-quality mp4

print('Done.')
