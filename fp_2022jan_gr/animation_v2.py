'''
Plot animation of agents in FPsim
'''

import os
import numpy as np
import sciris as sc
import pylab as pl
import fpsim as fp
import fp_analyses as fa


n_side   = 10
n        = n_side**2
filename = 'animation_data_v2.obj'
rerun    = 0
doplot   = 1
animate  = 1
overlay  = 0
dosave   = 1

if rerun or not os.path.exists(filename):

    data = sc.objdict()
    data.trim = dict()
    data.raw = dict()

    def record(sim, min_age=10, max_age=50):
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
            entry.boygirl  = sc.dcp(sim.people.children[:nrec])
            for p,childlist in enumerate(entry.boygirl):
                entry.boygirl[p] = sim.people.sex[childlist]
            age = sim.people.age[:nrec]
            entry.nonfecund = np.logical_not((age > min_age) * (age < max_age) * (sim.people.sex[:nrec] == 0))
            entry.active *= ~entry.nonfecund

            data[i][sim.i] = entry

        return

    pars = fa.senegal_parameters.make_pars()
    pars['n'] = n
    pars['start_year'] = 1990
    pars['end_year']   = 2021
    pars['interventions'] = [record]
    sim = fp.Sim(pars)
    sim.run()

    sim.plot()
    data.births = sim.results['births']
    data.deaths = sim.results['deaths']

    sc.save(filename, data)

else:

    data = sc.load(filename)


#%% Assemble people object
def count_ppl(entry):
    return len(entry.sex) # Can be anything, just get the length of the array

npts    = len(data.trim)
last_entry = data.raw[npts-1]
npeople = count_ppl(last_entry)
years   = np.array([data.raw[i].y for i in range(npts)])
paxmin, paxmax = years.min(), years.max()
ppl     = np.ones((npeople, npts, 3))


#%% Plot

if doplot:

    sc.options(font='Avenir')

    cmap = sc.objdict(
        nonfecund = '#dddddd',
        inactive  = '#bec7ff',
        active    = '#65c581',#'#63a763',
        preg      = '#d6e754',#'#1c8446',#'#e0e04d',
        method    = '#c43b3b',
        dead      = '#ffffff',
    )
    childcolor = '#aaaaaa'

    cmaparr = sc.objdict({k:sc.hex2rgb(v) for k,v in cmap.items()})

    cnames = sc.objdict(
        inactive  = 'Not sexually\nactive',
        active    = 'Sexually\nactive',
        preg      = 'Pregnant',
        method    = 'Using\ncontraception',
        nonfecund = 'Not fecund',
        dead      = 'Dead',
    )

    max_children = 9
    max_method = 10
    radius = 0.25
    mothersize = 100
    childsize = 10

    x = np.arange(n_side)
    y = np.arange(n_side)
    xx,yy = np.meshgrid(x,y)
    xx = xx.flatten()
    yy = yy.flatten()

    sc.options(dpi=200)
    fignum = 'Animation'
    fig = pl.figure(fignum, figsize=(15,6))

    # Animation axes
    ax = pl.axes([0,0,0.5,1])
    δ = 0.5
    pl.axis('off')

    # Plot axes
    pax = pl.axes([0.58,0.1,0.4,0.8])

    frames = []

    def xtr(x):
        return 0.05+x/(n_side+3.5)

    def ytr(y):
        return 0.05+y/(n_side+1)

    stride = 1 # Don't render every frame
    h_axvl = None # Create reference for updating
    h_img = None
    with sc.timer('generating'):
        print('Generating...')
        anim = sc.animation(fig=fig)
        for e in range(npts)[::stride]:
            if not pl.fignum_exists(fignum):
                raise Exception('Figure closed')

            #%% Preliminaries
            entry = data.trim[e]
            r_ent = data.raw[e]
            print(f'  Working on {entry.i} of {npts}...')
            frame = sc.autolist()
            ax.clear()
            pax.clear()

            # LHS -- animation

            # Handle counts
            counts = sc.objdict()
            f = ~entry.sex
            m = entry.sex
            alive = ~entry.dead
            alive_women = f * alive

            counts.nonfecund = entry.nonfecund[alive].sum()
            counts.active   = entry.active[alive_women].sum()
            counts.inactive = alive_women.sum() - counts.active
            counts.preg     = entry.preg[alive_women].sum()
            counts.method   = (entry.method[alive_women]>0).sum()
            counts.dead     = entry.dead.sum()

            cc = np.array([cmap.inactive]*n, dtype=object)
            n_raw = count_ppl(r_ent)
            ccarr = np.array([cmaparr.inactive]*n_raw)
            colorkeys = [k for k in cmap.keys() if k != 'inactive']
            newi = sc.objdict()
            newkeys = ['preg', 'method']
            for key in colorkeys:
                inds   = sc.findinds(entry[key])
                r_inds = sc.findinds(r_ent[key])
                cc[inds] = cmap[key]
                ccarr[r_inds,:] = cmaparr[key]
                if key in newkeys and e:
                    prev = ppl[r_inds, e-1, :]
                    this = ccarr[r_inds, :]
                    changed = np.logical_not((prev == this).prod(axis=1))
                    newi[key] = r_inds[changed]

            ppl[:n_raw, e, :] = ccarr # Copy into people object

            percents = sc.objdict()
            percents.nonfecund = sc.safedivide(counts.nonfecund, alive.sum()) * 100
            percents.inactive  = sc.safedivide(counts.inactive, alive_women.sum()) * 100
            percents.active    = sc.safedivide(counts.active, alive_women.sum()) * 100
            percents.preg      = sc.safedivide(counts.preg, alive_women.sum()) * 100
            percents.method    = sc.safedivide(counts.method, alive_women.sum()) * 100
            percents.dead      = sc.safedivide(counts.dead, n) * 100
            ave_age = np.median(entry.age[alive_women])

            # Plot legend -- have to do manually since legend not supported by animation
            dy = 0.7
            for i,key,label in cnames.enumitems():
                kwargs = dict(horizontalalignment='left', verticalalignment='center')
                x = n_side + 0.2
                y = n_side - 2.4 - dy*i
                x2 = x + 0.25
                y2 = y - 0.05
                lw = 0.1 if key == 'dead' else 0
                frame += ax.scatter(xtr(x), ytr(y), s=mothersize, c=cmap[key], linewidths=lw, edgecolor='k')
                frame += ax.text(xtr(x2), ytr(y2), f'{label} ({percents[key]:0.0f}%)', **kwargs)

            # Define markers
            fmark = 'o' # Female
            mmark = 's' # Male

            # Legend
            y3 = y2 + 5.0
            y4 = y3 - dy/2
            y5 = y4 - dy/2
            frame += ax.scatter(xtr(x), ytr(y3), s=mothersize, c='k', marker=fmark)
            frame += ax.scatter(xtr(x), ytr(y4), s=mothersize, c='k', marker=mmark)
            frame += ax.scatter(xtr(x), ytr(y5), s=childsize, c='k', marker=fmark)
            frame += ax.text(xtr(x2), ytr(y3), 'Woman', **kwargs)
            frame += ax.text(xtr(x2), ytr(y4), 'Man', **kwargs)
            frame += ax.text(xtr(x2), ytr(y5), 'Child', **kwargs)

            # Actually plot
            def age2size(ages, which='adult'):
                if len(ages):
                    if which == 'adult':
                        base = mothersize*0.5
                        factor = 10
                    else:
                        base = childsize*0.2
                        factor = 2
                    return base + factor*np.sqrt(ages)
                else:
                    return []

            fsize = age2size(entry.age[f])
            msize = age2size(entry.age[m])
            # print(mothersize, fsize, msize)
            frame += ax.scatter(xtr(xx[f]), ytr(yy[f]), s=fsize, c=cc[f], marker=fmark)
            frame += ax.scatter(xtr(xx[m]), ytr(yy[m]), s=msize, c=cc[m], marker=mmark)
            for m, (ix,iy) in enumerate(zip(xx,yy)):
                n_children = len(entry.children[m])
                if n_children:
                    c_arr = np.arange(n_children)
                    rad = 2*np.pi*c_arr/max_children - np.pi/2
                    dx = radius*np.cos(rad)
                    dy = radius*np.sin(rad)
                    girls = sc.findinds(entry.boygirl[m]==0)
                    boys  = sc.findinds(entry.boygirl[m]==1)
                    g_inds = r_ent.age[np.array(r_ent.children[m])[girls]] if len(girls) else []
                    b_inds = r_ent.age[np.array(r_ent.children[m])[boys]]  if len(boys)  else []
                    for inds,r_inds,mark in zip([girls, boys], [g_inds, b_inds], [fmark, mmark]):
                        frame += ax.scatter(xtr(ix+dx[inds]), ytr(iy+dy[inds]), s=age2size(r_inds, 'child'), c=childcolor, marker=mark)

            kwargs = dict(horizontalalignment='center', fontsize=12) # Set the "title" properties
            frame += ax.text(0.4, 0.93, f'Year {entry.y:0.0f}\nMedian age of cohort {ave_age:0.1f}', **kwargs) # Unfortunately pl.title() can't be dynamically updated
            frames.append(frame)
            ax.set_xlim((0, 1))
            ax.set_ylim((0, 1))
            ax.axis('off')

            #%% RHS -- cumulative sim
            if e:

                # People
                if h_img: h_img.remove()
                if h_axvl: h_axvl.remove()
                img    = ppl[:, 0:e, :]
                h_img  = pax.imshow(img, origin='lower', aspect='auto',  extent=[years[0], years[e], 0, npeople])
                h_axvl = pax.axvline(years[e], c='#bbb', lw=1)
                for key in newkeys:
                    dots = newi[key]
                    pax.scatter([years[e]]*len(dots), dots, c=cmap[key])


                # Births and deaths
                if overlay:

                    cum_births = np.cumsum(data.births[:e])
                    cum_deaths = np.cumsum(data.deaths[:e])
                    x = years[:e]
                    pax.plot(x, cum_births, c='green')
                    pax.plot(x, cum_deaths, c='k')


            #%% Tidying
            pax.set_xlim((paxmin, paxmax))
            pax.set_ylim((0, npeople))
            pax.set_xlabel('Year')
            pax.set_ylabel('Count')
            ticktop = False
            if ticktop:
                pax.xaxis.tick_top()
                pax.xaxis.set_label_position('top')
                sc.boxoff(pax, which=['right','bottom'])
            else:
                sc.boxoff(pax)

            if dosave:
                anim.addframe()
            if animate:
                pl.pause(0.01)

    if dosave:
        with sc.timer('saving'):
            print('Saving...')
            sc.runcommand('ffmpeg -r 10 -i animation_%04d.png -q:v 1 fp_animation_schematic_v2.mp4')
            # anim.save('fp_animation_schematic_v2.mp4', fps=10, tidy=False) # SLOW AND BROKEN

print('Done.')
