import fpsim as fp
import optuna
import sciris as sc
import numpy as np

# Define upper and lower limits for the parameters
#ranges = dict(
    #space0_9=[0.5, 1],
    #space12=[0.5, 1.5],
    #space15=[1.0, 2.0],
    #space15_36=[1.0, 10.0]
#)

ranges = {
    '0' : [0.2, 1],
    '3' : [0.2, 1],
    '6' : [0.2, 1],
    '9' : [0.2, 1],
    '12': [0.5, 1.2],
    '15': [0.5, 1.2],
    '18': [1.0, 5.0],
    '21': [1.0, 5.0],
    '24': [1.0, 5.0],
    '27': [1.0, 5.0],
    '30': [1.0, 5.0],
    '33': [1.0, 5.0],
    '36': [1.0, 5.0]
}



# Birth spacing percent data
# data = [16.29, 60.12, 22.29, 1.30]

spacing_file = 'BirthSpacing.obj'

spacing_bins = sc.odict({'0-12': 0, '12-24': 1, '24-48': 2, '>48': 4})  # Spacing bins in years
min_age = 15
max_age = 50

# Load birth spacing data
data = sc.load(spacing_file)


def make_sim(spacing_pars):
    base_pars = dict(location='senegal')
    prefs = np.array(list(spacing_pars.values()))
    print(f'spacing pars values: {prefs}')
    sim = fp.Sim(base_pars)
    sim.pars['spacing_pref']['preference'] = prefs
    return sim

def get_data_spaces(data):
    # Extract birth spacing data from  data
    spacing = data['spacing']  # Extract from data
    data_spacing_counts = sc.odict().make(keys=spacing_bins.keys(), vals=0.0)

    # Spacing bins from data
    spacing_bins_array = sc.cat(spacing_bins[:], np.inf)
    for i in range(len(spacing_bins_array) - 1):
        lower = spacing_bins_array[i]
        upper = spacing_bins_array[i + 1]
        matches = np.intersect1d(sc.findinds(spacing >= lower), sc.findinds(spacing < upper))
        data_spacing_counts[i] += len(matches)

    spaces = data_spacing_counts.values()
    return spaces

def get_spaces(sim):
    model_spacing_counts = sc.odict().make(keys=spacing_bins.keys(), vals=0.0)
    sim.run()
    ppl = sim.people
    for i in range(len(ppl)):
        if ppl.alive[i] and not ppl.sex[i] and ppl.age[i] >= min_age and ppl.age[i] < max_age:
            if len(ppl.dobs[i]) > 1:
                for d in range(len(ppl.dobs[i]) - 1):
                    space = ppl.dobs[i][d + 1] - ppl.dobs[i][d]
                    ind = sc.findinds(space > spacing_bins[:])[-1]
                    model_spacing_counts[ind] += 1

    spaces = model_spacing_counts.values()
    return spaces


def compute_mismatch(data_spaces, sim_spaces):
    diffs = np.array(data_spaces) - np.array(sim_spaces)
    mismatch = sum(diffs ** 2)
    return mismatch


def run_sim(trial):
    spacing_pars = dict()
    for key, (vmin, vmax) in ranges.items():
        spacing_pars[key] = trial.suggest_float(key, vmin, vmax)

    sim = make_sim(spacing_pars)
    sim_spaces = get_spaces(sim)
    data_spaces = get_data_spaces(data)
    mismatch = compute_mismatch(data_spaces, sim_spaces)
    return mismatch


if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(run_sim, n_trials=10)

    print(study.best_params)