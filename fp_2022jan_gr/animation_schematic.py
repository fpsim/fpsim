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
δ = 0.5
pl.xlim((-δ, n_side-1+δ))
pl.ylim((-δ, n_side-1+δ))
sc.figlayout()

frames = []

        # for i in range(nframes): # Loop over the frames
        #     dots += pl.randn(ndots, 2) # Move the dots randomly
        #     color = pl.norm(dots, axis=1) # Set the dot color
        #     old = pl.array(old_dots) # Turn into an array
        #     plot1 = pl.scatter(old[:,0], old[:,1], c='k') # Plot old dots in black
        #     plot2 = pl.scatter(dots[:,0], dots[:,1], c=color) # Note: Frames will be separate in the animation
        #     pl.xlim((-axislim, axislim)) # Set x-axis limits
        #     pl.ylim((-axislim, axislim)) # Set y-axis limits
        #     kwargs = {'transform':pl.gca().transAxes, 'horizontalalignment':'center'} # Set the "title" properties
        #     title = pl.text(0.5, 1.05, f'Iteration {i+1}/{nframes}', **kwargs) # Unfortunately pl.title() can't be dynamically updated
        #     pl.xlabel('Latitude') # But static labels are fine
        #     pl.ylabel('Longitude') # Ditto
        #     frames.append((plot1, plot2, title)) # Store updated artists
        #     old_dots = pl.vstack([old_dots, dots]) # Store the new dots as old dots
        # sc.savemovie(frames, 'fleeing_dots.mp4', fps=20, quality='high') # Save movie as a high-quality mp4

with sc.timer('generating'):
    print('Generating...')
    for entry in list(data.values()):
        print(f'  Working on {entry.i} of {len(data)}...')
        frame = sc.autolist()
        # pl.cla()
        # cc = sc.vectocolor(entry.method[:n], maxval=max_method)
        cc = entry.method[:n] > 0
        frame += pl.scatter(xx, yy, s=100, c=cc, marker='o')
        for m, (ix,iy) in enumerate(zip(xx,yy)):
            n_children = len(entry.children[m])
            if n_children:
                c_arr = np.arange(n_children)
                rad = 2*np.pi*c_arr/max_children
                dx = radius*np.cos(rad)
                dy = radius*np.sin(rad)
                frame += pl.scatter(ix+dx, iy+dy, s=10, c='k')

        kwargs = {'transform':pl.gca().transAxes, 'horizontalalignment':'center'} # Set the "title" properties
        frame += pl.text(0.5, 1.00, f'{entry.y:0.1f}', **kwargs) # Unfortunately pl.title() can't be dynamically updated
        frames.append(frame)
        pl.axis('off')
        # pl.pause(0.01)

with sc.timer('saving'):
    print('Saving...')
    sc.savemovie(frames, 'fp_animation_schematic.mp4', fps=30, quality='high') # Save movie as a high-quality mp4

print('Done.')
