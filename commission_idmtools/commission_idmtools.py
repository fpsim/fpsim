import random

from GenerateCampaignRCM import * # TODO: refactor
from idmtools.builders import ExperimentBuilder
from idmtools.core.platform_factory import Platform
from idmtools.managers import ExperimentManager
from idmtools_model_emod.emod_experiment import EMODExperiment

exp_name = 'Family Planning Workflow Development'
config_fn = os.path.join('inputs', 'fp_default_config.json')

# Commonly modified calibration variables and configuration
BASE_POPULATION_SCALE_FACTOR = 0.00333333333333333  # For quick test simulations, this is set to a very low value
N_SAMPLES = 3  # the number of distinct parameter sets to run per iteration
N_REPLICATES = 1  # replicates, 1 is highly recommended.

samples = pd.DataFrame({'PillEfficacy': np.linspace(0, 1, N_SAMPLES)})

static_params = {
    'Base_Population_Scale_Factor': BASE_POPULATION_SCALE_FACTOR,
    'Custom_Individual_Events': [
        'Should_Not_Be_Broadcasted',
        'Choose_Next_Method_Currently_Under_Age',
        'Choose_Next_Method_Currently_On_Implant',
        'Choose_Next_Method_Currently_On_Pill',
        'Choose_Next_Method_Currently_On_Withdrawal',
        'Choose_Next_Method_Currently_Pregnant',
        'Choose_Next_Method_Currently_Post_Partum',
        'Choose_Next_Method_Currently_On_None',
        'Use_Implant',
        'Use_Pill',
        'Use_Withdrawal',
        'Use_None',
    ],
    'Enable_Property_Output': 1,
    'Report_Event_Recorder': 1,
    'Report_Event_Recorder_Events': [
        'Pregnant',
        'GaveBirth',
        'Choose_Next_Method_Currently_Under_Age',
        'Choose_Next_Method_Currently_On_Implant',
        'Choose_Next_Method_Currently_On_Pill',
        'Choose_Next_Method_Currently_On_Withdrawal',
        'Choose_Next_Method_Currently_Pregnant',
        'Choose_Next_Method_Currently_Post_Partum',
        'Choose_Next_Method_Currently_On_None',
        'Use_Implant',
        'Use_Pill',
        'Use_Withdrawal',
        'Use_None',
    ],
    'Report_Event_Recorder_Ignore_Events_In_List': 0,
    'Report_Event_Recorder_Individual_Properties': [
        'CurrentStatus'
    ],
    'logLevel_RandomChoiceMatrix': 'ERROR'
}


def map_sample_to_model_input_fn(simulation, sample_dict):
    con_list = CreateContraceptives()

    pill_contraceptive = next(x[1] for x in con_list if x[0] == USE_PILL)
    pill_contraceptive.Waning_Config.Initial_Effect = sample_dict['PillEfficacy']

    rc_list = CreateRandomChoiceMatrixList()
    campaign = GenerateCampaignFP(con_list, rc_list)

    simulation.campaign = json.loads(campaign.to_json())

    sample_index = sample_dict.pop("sample_index")
    return {"sample_index": sample_index, **{'[SAMPLE] %s' % k: v for k, v in sample_dict.items()}}


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
