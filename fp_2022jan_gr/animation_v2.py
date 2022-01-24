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
filename = 'animation_data_v2.obj'
rerun  = 0
doplot = 1
dosave = 0

if rerun or not os.path.exists(filename):

    data = sc.objdict()
    data.trim = dict()
    data.raw = dict()

    def record(sim):
        for i,nrec in enumerate([n, None]): # Either trim, or don't
            entry = sc.objdict()
            entry.i = sim.i
            entry.y = sim.y
            entry.sex      = sc.dcp(sim.people.sex[:nrec])
            entry.age      = sc.dcp(sim.people.age[:nrec])
            entry.dead     = sc.dcp(~sim.people.alive[:nrec])
            entry.active   = sc.dcp(sim.people.sexually_active[:nrec])
            entry.preg     = sc.dcp(sim.people.pregnant[:nrec])
            entry.method   = sc.dcp(sim.people.method[:nrec])
            entry.children = sc.dcp(sim.people.children[:nrec])
            for p,childlist in enumerate(entry.children):
                entry.children[p] = sim.people.sex[childlist]

            data[i][sim.i] = entry

        return

    pars = fa.senegal_parameters.make_pars()
    pars['n'] = n
    pars['start_year'] = 1990
    pars['interventions'] = [record]
    sim = fp.Sim(pars)
    sim.run()

    sim.plot()
    data.births = sim.results['births']
    data.deaths = sim.results['deaths']

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
    fig = pl.figure(figsize=(15,6))

    # Animation axes
    ax = pl.axes([0,0,0.5,1])
    δ = 0.5
    pl.axis('off')

    # Plot axes
    pax = pl.axes([0.55,0.1,0.4,0.8])

    frames = []

    def xtr(x):
        return 0.05+x/(n_side+3.5)

    def ytr(y):
        return 0.05+y/(n_side+1)

    stride = 2 # Don't render every frame
    axvl = None # Create reference for updating
    with sc.timer('generating'):
        print('Generating...')
        npts = len(data.trim)
        for e in range(npts)[::stride]:

            #%% Preliminaries
            entry = data.trim[e]
            print(f'  Working on {entry.i} of {npts}...')
            frame = sc.autolist()
            if not dosave:
                ax.clear()

            # LHS -- animation

            # Handle counts
            counts = sc.objdict()
            f = ~entry.sex
            m = entry.sex
            alive = ~entry.dead
            alive_women = f * alive

            counts.active   = entry.active[alive_women].sum()
            counts.inactive = alive_women.sum() - counts.active
            counts.preg     = entry.preg[alive_women].sum()
            counts.method   = (entry.method[alive_women]>0).sum()
            counts.dead     = entry.dead.sum()

            cc = np.array([cmap.inactive]*n, dtype=object)
            colorkeys = ['active', 'preg', 'method', 'dead']
            for key in colorkeys:
                inds = sc.findinds(entry[key])
                cc[inds] = cmap[key]

            percents = sc.objdict()
            percents.inactive = sc.safedivide(counts.inactive, alive_women.sum()) * 100
            percents.active   = sc.safedivide(counts.active, alive_women.sum()) * 100
            percents.preg     = sc.safedivide(counts.preg, alive_women.sum()) * 100
            percents.method   = sc.safedivide(counts.method, alive_women.sum()) * 100
            percents.dead     = sc.safedivide(counts.dead, n) * 100
            ave_age = np.median(entry.age[alive_women])

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

            # Define markers
            fmark = 'o'
            mmark = 's'
            def sexmark(val):
                out = fmark if val==0 else mmark
                return out

            # Legend
            y3 = y2 + 4.5
            y4 = y3 - dy/2
            y5 = y4 - dy/2
            frame += ax.scatter(xtr(x), ytr(y3), s=mothersize, c='k', marker=fmark)
            frame += ax.scatter(xtr(x), ytr(y4), s=mothersize, c='k', marker=mmark)
            frame += ax.scatter(xtr(x), ytr(y5), s=childsize, c='k', marker=fmark)
            frame += ax.text(xtr(x2), ytr(y3), 'Woman', **kwargs)
            frame += ax.text(xtr(x2), ytr(y4), 'Man', **kwargs)
            frame += ax.text(xtr(x2), ytr(y5), 'Child', **kwargs)

            # Actually plot
            frame += ax.scatter(xtr(xx[f]), ytr(yy[f]), s=mothersize, c=cc[f], marker=fmark)
            frame += ax.scatter(xtr(xx[m]), ytr(yy[m]), s=mothersize, c=cc[m], marker=fmark)
            for m, (ix,iy) in enumerate(zip(xx,yy)):
                n_children = len(entry.children[m])
                if n_children:
                    c_arr = np.arange(n_children)
                    rad = 2*np.pi*c_arr/max_children - np.pi/2
                    dx = radius*np.cos(rad)
                    dy = radius*np.sin(rad)
                    girls = sc.findinds(entry.children[m]==0)
                    boys  = sc.findinds(entry.children[m]==1)
                    for inds,mark in zip([girls, boys], [fmark, mmark]):
                        frame += ax.scatter(xtr(ix+dx[inds]), ytr(iy+dy[inds]), s=childsize, c='k', marker=mark)

            kwargs = dict(horizontalalignment='center', fontsize=12) # Set the "title" properties
            frame += ax.text(0.4, 0.93, f'Year {entry.y:0.0f}\nMedian age of cohort {ave_age:0.1f}', **kwargs) # Unfortunately pl.title() can't be dynamically updated
            frames.append(frame)
            ax.set_xlim((0, 1))
            ax.set_ylim((0, 1))
            ax.axis('off')

            #%% RHS -- cumulative sim
            raw = data.raw[e] # Raw entry
            years = np.array([data.raw[i].y for i in range(npts)])
            xmin, xmax = years.min(), years.max()
            last = data.raw[npts-1]
            npeople = len(last.sex)

            # People
            ppl = pl.rand(npts, npeople, 3)#np.full((npts, npeople), np.nan)
            pax.imshow(ppl, origin='lower', aspect='auto', extent=[xmin, xmax, 0, npeople])

            # Births and deaths
            if e:
                if axvl:
                    axvl.remove()
                cum_births = np.cumsum(data.births[:e])
                cum_deaths = np.cumsum(data.deaths[:e])
                x = years[:e]
                pax.plot(x, cum_births, c='green')
                pax.plot(x, cum_deaths, c='k')
                axvl = pax.axvline(x[-1], c='#bbb', lw=1)

            # # Tidying
            # pax.set_xlim((xmin, xmax))
            # pax.set_ylim((0, npeople))
            # pax.set_xlabel('Year')
            # pax.set_ylabel('Count')
            # sc.boxoff(pax)


            #%% Tidy up
            if not dosave:
                pl.pause(0.01)

    if dosave:
        with sc.timer('saving'):
            print('Saving...')
            sc.savemovie(frames, 'fp_animation_schematic.mp4', fps=10, quality='high') # Save movie as a high-quality mp4

print('Done.')
