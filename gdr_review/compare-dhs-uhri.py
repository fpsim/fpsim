import pylab as pl
import sciris as sc
import fp_data.DHS.calobj as co

sc.heading('Setting parameters...')

plot_dhs  = 1
plot_pies = 0
save_figs = 1
uhri_file = '../fp_data/UHRI/senegal_women.obj' # Not Windows friendly but simpler and neither is later code
dhs_file  = '../fp_data/DHS/senegal.cal'


#%% Loading data
sc.heading('Loading data...')
uhri = sc.loadobj(uhri_file)
dhs = co.load(dhs_file) # Create from saved data
dhs.make_results()

#%% Processing
sc.heading('Processing data...')

# TODO: copied from analyze_UHRI.py
method_mapping = {
    'No method': 'No method', # Unmet?  Not quite...

    'Daily pill': 'Short-lasting',
    'Male condom': 'Short-lasting',
    'Female condom': 'Short-lasting',

    'Injectables': 'Injections',

    'Implants': 'Long-lasting',
    'iucd': 'Long-lasting',
    'Female sterilization': 'Long-lasting',

    # Some of these could be unmet?
    'sdm': 'Other',
    'Natural methods': 'Other',
    'Breastfeeding/LAM': 'Other',
    'Other traditional method': 'Other',
    'Emergency pill': 'Other',
    'Other modern method': 'Other',
}
uhri['MethodClass'] = uhri['Method'].replace(method_mapping)

# TODO: copied from calobj.py
orig_dhs_mapping = sc.odict({
         ' ': [-1, 'Miss',  'Missing', ''],
         'B': [ 0, 'Birth', 'Birth', ''],
         'T': [ 1, 'Term',  'Termination', ''],
         'P': [ 2, 'Preg',  'Pregnancy', ''],
         '0': [ 3, 'None',  'Non-use', 'No method'],
         '1': [ 4, 'Pill',  'Pill', 'Short-lasting'],
         '2': [ 5, 'IUD',   'IUD', 'Long-lasting'],
         '3': [ 6, 'Injct', 'Injectables', 'Injections'],
         '4': [ 7, 'Diaph', 'Diaphragm', 'Short-lasting'],
         '5': [ 8, 'Cond',  'Condom', 'Short-lasting'],
         '6': [ 9, 'FSter', 'Female sterilization', 'Long-lasting'],
         '7': [10, 'MSter', 'Male sterilization', 'Long-lasting'],
         '8': [11, 'Rhyth', 'Rhythm', 'Other'],
         '9': [12, 'Withd', 'Withdrawal', 'Other'],
         'W': [13, 'OTrad', 'Other traditional', 'Other'],
         'N': [14, 'Impla', 'Implants', 'Long-lasting'],
         'A': [15, 'Abst', 'Abstinence', 'No method'],
         'L': [16, 'Lact',  'Lactational amenorrhea', 'Other'],
         'C': [17, 'FCond', 'Female condom', 'Short-lasting'],
         'F': [18, 'Foam',  'Foam and jelly', 'Short-lasting'],
         'E': [19, 'Emerg', 'Emergency contraception', 'Other'],
         'S': [20, 'SDays', 'Standard days', 'Other'],
         'M': [21, 'OModr', 'Other modern', 'Other'],
        })

# Remap the above dictionary so the number is the key and the method class is the value
dhs_mapping = {}
for entry in orig_dhs_mapping.values():
    dhs_mapping[entry[0]] = entry[-1]

# Calculate the proportions
prop_data = sc.odict()
prop_data['UHRI'] = sc.odict(uhri['MethodClass'].value_counts().to_dict())
method_classes = prop_data['UHRI'].keys()
n_classes = len(method_classes)
method_colors = sc.gridcolors(ncolors=n_classes)
prop_data['DHS'] = sc.odict().make(keys=method_classes, vals=0)
index = -1 # Use only the most recent time point
dhs_cal_data = dhs.cal[:,index]
for val in dhs_cal_data:
    key = dhs_mapping[val]
    if key: # Skip '', which is excluded in UHRI
        prop_data['DHS'][key] += 1

#%% Plotting
sc.heading('Plotting...')

if plot_dhs:
    f1 = dhs.plot_transitions()
    f2 = dhs.plot_transitions(projection='3d')
    f3 = dhs.plot_slice('None', stacking='sidebyside')
    if save_figs:
        f1.savefig('figs/dhs_transitions_2d.png', bbox_inches='tight')
        f2.savefig('figs/dhs_transitions_3d.png', bbox_inches='tight')
        f3.savefig('figs/dhs_transitions_slice.png', bbox_inches='tight')

if plot_pies:
    keys1 = ['UHRI', 'DHS']
    keys2 = ['All women', 'Women on a method']
    fig = pl.figure(figsize=(24,18))
    axs = sc.odict()
    off = 0.05
    wid = 0.45
    hei = 0.45
    for k1,key1 in enumerate(keys1):
        for k2,key2 in enumerate(keys2):
            ax = fig.add_axes([off+wid*k1, (off+hei)*(1-k2), wid, hei])
            data = prop_data[key1][k2:]
            labels = [f'{key} (n={val})' for key,val in prop_data[key1].items()][k2:]
            colors = method_colors[k2:]
            explode = 0.02*pl.ones(n_classes)[k2:]
            _, _, autotexts = ax.pie(data, labels=labels, colors=colors, explode=explode, autopct='%0.1f%%')
            for autotext in autotexts:
                autotext.set_color('white')
            ax.set_title(f'{key1} data: {key2}', fontweight='bold')
            axs[key1+key2] = ax
    if save_figs:
        pl.savefig('figs/uhri_vs_dhs_pies.png', bbox_inches='tight')
    
            

print('Done.')