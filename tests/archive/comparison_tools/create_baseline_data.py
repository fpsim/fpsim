from collections import defaultdict
import sys
import argparse
import os
import pandas as pd
import numpy as np
import fpsim as fp
import fp_analyses as fa


timeseries_comparison = ["alive"]
end_of_sim_comparison = ["pop_size"]
sweep = True
scenario_parameters = {"n": [500, 1000, 1500]} # Leave empty dict if none
if sweep:
    number_of_sims = len(scenario_parameters[list(scenario_parameters.keys())[0]]) # getting length of sweep values
else:
    number_of_sims = 2

def run_sims():
    """
    Runs "number_of_sims" sims as a MultiSim object and outputs the object
    """
    sim_list = [0] * number_of_sims
    pars = fa.senegal_parameters.make_pars()
    pars['n'] = 1000

    for i in range(number_of_sims):
        if sweep:
            for variable in scenario_parameters:
                pars.update({variable: scenario_parameters[variable][i]})
            pars['seed'] = i
            print(f"Seed: {i}, pop: {pars[variable]}")
        sim_list[i] = fp.SimVerbose(pars)

    suite = fp.MultiSim(sims=sim_list)
    suite.run() 
    return suite

def create_baseline(suite, baseline_file = "comparison_data/baseline_results.csv"):
    """
    Creates a new baseline .csv file at the specified location
    Args:
        baseline_file::str
            The destination for the created baseline file
    """
    results = create_multi(suite, version="master")

    df = pd.DataFrame(results)

    df.to_csv(baseline_file, index=False)
    print(f"Created new baseline file at {baseline_file}")

def create_multi(suite, version="new"):
    """
    Runs "number_of_sims" sims and saves the results as a dictionary to be formatted as a dataframe
    Args:
        version::str
            A label signifying the version of the sim that the data is extracted from
    
    Notes:
        The produced dictionary has keys representing columns [version, seed, variable, value]:
            version::str:
                version of sim from which the data has been extracted ["new" | "master"]
            seed::int:
                seed used when creating corresponding sim
            variable::str:
                variable extracted at end of sim (see bottom of file for enumeration)
            value::int:
                value of variable at end of sim
    """
    intermediate_results = defaultdict(list)
    for sim in suite.sims:
        for variable in end_of_sim_comparison:
            intermediate_results[variable].append(sim.results[variable][-1])

    # intermediate results are formatted as: {num_infections: [50, 40, 50, ...]}

    results = {'version': [], 'seed': [], 'variable': [], 'value': []}

    for variable in intermediate_results:
        for index, value in enumerate(intermediate_results[variable]):
            results['version'].append(version)
            results['variable'].append(variable)
            results['seed'].append(index)
            results['value'].append(value)

    return results

def compare_results(suite, baseline_file="comparison_data/baseline_results.csv", destination_file="comparison_data/final.csv"):
    """
    Runs "number_of_sims" sims, extracts data, and combines this data with data from baseline_file to destination_file
    Args:
        baseline_file::str
            The location of the baseline file
        destination_file::str
            The location of the ouput file combining new and baseline data
    Outputs:
        File at destination_file formatted as specified in create_multi()
    """
    results = create_multi(suite)
    baseline_df = pd.read_csv(baseline_file)
    for key in results:
        for item in baseline_df[key]:
            results[key].append(item)

    final_df = pd.DataFrame(results)
    final_df.to_csv(destination_file, index=False)
    print(f"Combined baseline results and new results in {destination_file}")

def replace_baseline(baseline_file="comparison_data/baseline_results.csv", results_file="comparison_data/final.csv"):
    """
    Replaces data in baseline_file with the new data filtered from results_file
    Args:
        baseline_file::str
            The location of the output baseline file
        results_file::str
            The location of the file containing the new baseline data to be filtered
    Outputs:
        File at baseline_file formatted as specified in create_multi()
    """
    total_data = pd.read_csv(results_file)
    total_data = total_data[total_data["version"] == "new"]
    total_data.to_csv(baseline_file, index=False)
    print(f"Replaced baseline at {baseline_file} with new results from {results_file}")

def create_multi_timeseries(suite, version="new"):
    """
    Runs "number_of_sims" sims and saves the results as a dictionary to be formatted as a dataframe
    Args:
        version::str
            A label signifying the version of the sim that the data is extracted from
    Notes:
        The produced dictionary has keys representing columns [version, seed, variable, value]:
            version::str:
                version of sim from which the data has been extracted ["new" | "master"]
            seed::int:
                seed used when creating corresponding sim
            variable::str:
                variable extracted at end of sim (see bottom of file for enumeration)
            value::int:
    """

    final_results = {'version': [], 'seed': [], 'variable': [], 'time': [], 'value': [], 'inf': []}
    last_value = 0
    for sim_index, sim in enumerate(suite.sims):
        results = sim.total_results
        for timestep in results:
            for variable in timeseries_comparison:
                value = sum(results[timestep][variable])
                final_results['version'].append(version)
                final_results['seed'].append(sim_index)
                final_results['variable'].append(variable)
                final_results['time'].append(timestep)
                final_results['value'].append(value)
                final_results['inf'].append(value - last_value)

                last_value = value

    return final_results

def create_baseline_timeseries(suite, baseline_file = "baseline_results_timeseries.csv"):
    """
    Creates a new baseline .csv file at the specified location
    Args:
        baseline_file::str
            The destination for the created baseline file
    """
    results = create_multi_timeseries(suite, version="master")

    df = pd.DataFrame(results)

    df.to_csv(baseline_file, index=False)
    print(f"Created new baseline file at {baseline_file}")

def compare_results_timeseries(suite, baseline_file="baseline_results_timeseries.csv", destination_file="final_timeseries.csv"):
    """
    Runs "number_of_sims" sims, extracts data, and combines this data with data from baseline_file to destination_file
    Args:
        baseline_file::str
            The location of the baseline file for the timeseries analysis
        destination_file::str
            The location of the ouput file combining new and baseline data for timeseries
    Outputs:
        File at destination_file formatted as specified in create_multi()
    """
    results = create_multi_timeseries(suite)
    baseline_df = pd.read_csv(baseline_file)
    for key in results:
        for item in baseline_df[key]:
            results[key].append(item)

    final_df = pd.DataFrame(results)
    final_df.to_csv(destination_file, index=False)
    print(f"Combined baseline results and new results in {destination_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--new', help='Create a new baseline file', required=False, action='store_true')
    parser.add_argument('-c', '--compare', help='Create table with both baseline (baseline_results.csv) and new results ', action='store_true')
    parser.add_argument('-res', '--replace', help='Replace the baseline file with the new results input', action='store_true')

    if not os.path.exists("comparison_data"):
        os.mkdir("comparison_data")
    baseline_file = "comparison_data/baseline_results.csv"
    result_file = "comparison_data/data_to_plot.csv"

    baseline_file_ts = "comparison_data/baseline_results_ts.csv"
    result_file_ts = "comparison_data/data_to_plot_ts.csv"

    args = parser.parse_args()

    if args.replace:
        replace_baseline(baseline_file_ts, result_file_ts)
        replace_baseline(baseline_file, result_file)
    else:
        suite = run_sims()
        if args.new:
            create_baseline_timeseries(suite, baseline_file_ts)
            create_baseline(suite, baseline_file)

        if args.compare:
            compare_results_timeseries(suite, baseline_file_ts, result_file_ts)
            compare_results(suite, baseline_file, result_file)
