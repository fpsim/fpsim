import os
import numpy as np
import pyemod as em

# This should be emod_api
import json
import GenerateCampaignRCM as gencam

exp_name = 'Family Planning Workflow Development'
inputs = 'inputs'
config_file = os.path.join(inputs, 'config.json') # TODO: Remove boilerplate
demographics_file = os.path.join(inputs, 'demographics.json')

# Commonly modified calibration variables and configuration
BASE_POPULATION_SCALE_FACTOR = 0.0033  # For quick test simulations, this is set to a very low value
N_SAMPLES = 3  # the number of distinct parameter sets to run per iteration
N_REPLICATES = 1  # replicates, 1 is highly recommended.

samples = np.linspace(0, 1, N_SAMPLES)

base_sim= em.Simulation(config=config_file, demographics=demographics_file)

def make_campaign(pill_efficacy):
    campaign = em.default_campaign()
    con_list = gencam.CreateContraceptives()
    pill_contraceptive = next(x[1] for x in con_list if x[0] == gencam.USE_PILL)
    pill_contraceptive.Waning_Config.Initial_Effect = pill_efficacy
    rc_list = gencam.CreateRandomChoiceMatrixList()
    campaign = gencam.GenerateCampaignFP(con_list, rc_list)
    campaign.pars = json.loads(campaign.to_json())
    return campaign

for value in samples:
    base_sim.campaign = make_campaign(pill_efficacy=value)






def change_seed(simulation, seed):
    simulation.set_parameter("Run_Number", seed)
    return {"seed": seed}


if __name__ == "__main__":
    # Choose where to run
    platform = Platform("COMPS")

    # Prepare the path for the exe and demographics needed
    eradication_path = os.path.join("bin", "Eradication_FPOngoing_a4be1893d72d4df3de568217ecc63ecfcaee17cf.exe")
    demographics_path = [os.path.join("inputs", "demographics.json"), os.path.join("inputs", "IP_Overlay.json")]

    # Create the experiment from_files
    experiment = EMODExperiment.from_files(name=exp_name,
                                           eradication_path=eradication_path,
                                           config_path=config_fn,
                                           demographics_paths=demographics_path)

    # Update the basic parameters in the base_simulation of the experiment
    experiment.base_simulation.update_parameters(static_params)

    # Create the sweep
    builder = ExperimentBuilder()
    builder.add_sweep_definition(change_seed, [random.randint(0, 65535) for _ in range(N_REPLICATES)])

    # Sweep on the samples
    sample_dicts = [
        {"sample_index": sample[0], **{k: v for k, v in zip(samples.columns.values, sample[1:])}}
        for sample in samples.itertuples()
    ]
    builder.add_sweep_definition(map_sample_to_model_input_fn, sample_dicts)

    # Associate the builder to the experiment
    experiment.add_builder(builder)

    # Running
    manager = ExperimentManager(experiment=experiment, platform=platform)
    manager.run()
    manager.wait_till_done()
