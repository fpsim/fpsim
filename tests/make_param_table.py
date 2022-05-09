import sys
import argparse
import fpsim as fp
import fp_analyses as fa
import pandas as pd
import numpy as np

def run_experiment():
    """
    Creates ExperimentVerbose object and runs it with default parameters

    Args:
        None
    Returns:
        ExperimentVerbose object post running simulation
    """
    pars = fa.senegal_parameters.make_pars()
    exp = fp.ExperimentVerbose(pars)
    exp.run(keep_people = True)
    return exp

def cum_sum_none(series):
    """
    Utility function like np.cumsum() but skips over None values
    Args:
        series::pd.Series:
            pd.Series object representing an individual's state over time
    Returns:
        pd.Series object representing an individual's cumulative state over time
    """
    total = 0
    result_list = []
    for item in series:
        if item is not None:
            total = total + item
        result_list.append(total)
    return pd.Series(result_list)

def save_param_as_csv(exp, param, cumulative=False, min_age=20, max_age=25, sparse=True):
    """
    Args:
        exp::ExperimentVerbose:
            ExperimentVerbose object which has been run
        param::str:
            Name of parameter to save to csv. Check defaults.debug_states for enumerative list
        cumulative::bool:
            True or False if the result per person should be calculated cumulatively
        min_age::int:
            Minimum age that the individual must be for their results to be recorded. A person
            who becomes this age will start to have their results recorded
        max_age::int:
            Maximum age that the individual can be for their results to be recorded. A person
            who exceeds this age will stop having their results recorded
        sparse::bool:
            If True, the saved csv will only contain the rows which have at least one nonzero 
            value
    Output:
        A csv file saved at sim_output/[param]_timeseries.csv with columns being timesteps and 
        rows representing individual states over time for specified parameter

    """
    result_dict = exp.total_results

    dataframe_dict = {}
    for timestep in result_dict:
        result_list = [None] * len(result_dict[timestep][param])
        for index, param_value in enumerate(result_dict[timestep][param]):
            person_age = result_dict[timestep]["age"][index]
            if person_age >= min_age and person_age <= max_age:
                result_list[index] = param_value
            dataframe_dict[timestep] = result_list

    max_length = len(result_dict[max(result_dict.keys())][param])

    # Since there is a growing number of rows (people) per timestep we need to adjust for consistency
    for timestep in dataframe_dict:
        adjustment = max_length - len(dataframe_dict[timestep])
        dataframe_dict[timestep] = list(dataframe_dict[timestep]) + [None] * adjustment

    dataframe = pd.DataFrame(dataframe_dict)
    if sparse:
        dataframe = dataframe.loc[~(dataframe == 0).all(axis=1)]

    if cumulative:
        dataframe = dataframe.cumsum(axis=1)

    dataframe.to_csv(f"sim_output/{param}_timeseries.csv")
    print(f"Saved table to sim_output/{param}_timeseries.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--parameter', help='Parameter you want to print a csv of', required=True)
    parser.add_argument('-c', '--cumulative', help='Create csv file with each row value being cumulative', required=False, action='store_true')
    parser.add_argument('-g', '--greaterthan', help='Only record data of individuals while their age is greater than this value', required=False, default=0)
    parser.add_argument('-l', '--lessthan', help='Only record data of individuals while their age is less than this value', required=False, default=100)
    parser.add_argument('-s', '--sparse', help='Only record rows (individuals) with at least one non-zero value', required=False, action='store_true')

    args = parser.parse_args()

    experiment = run_experiment()
    save_param_as_csv(experiment, args.parameter, args.cumulative, args.greaterthan, args.lessthan, args.sparse)


# possible keys: ["alive", "breastfeed_dur", "gestation", "lactating", 
#                 "lam", "postpartum", "pregnant", "sexually_active",
#                 "postpartum_dur", "parity", "method", "age"]