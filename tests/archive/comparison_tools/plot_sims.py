import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

baseline_file = os.path.join(os.getcwd(), "comparison_data", "data_to_plot.csv")
baseline_file_ts = os.path.join(os.getcwd(), "comparison_data", "data_to_plot_ts.csv")

def plot_baseline_comparison(colorblind=False, results_file=baseline_file):
    """
    Plots and shows a boxplot comparing the baseline results to the new sim results
    Args:
        colorblind::bool:
            If this is False the color scheme is the default in seaborn otherwise it is the
            colorblind friendly ['blue', 'yellow']
        results_file::str:
            Destination of the csv file being plotted
    Outputs:
        Faceted boxplots rendered via matplotlib.pyplot
    """
    data = pd.read_csv(results_file)

    palette = None
    if colorblind:
        palette = ['blue', 'yellow']

    sns.catplot(
        data=data, x='version', y='value',
        col='variable', kind='box', col_wrap=2,
        order=['master', 'new'],
        palette=palette,
    )
    plt.show()


def plot_timeseries_comparison(results_file=baseline_file_ts):
    """
    Plots and shows a timeseries ribbon plot and trace plot comparing the baseline results to the new sim results
    Args:
        results_file::str:
            Destination of the csv file being plotted
    Outputs:
        Ribbon plot & line plot rendered via matplotlib.pyplot
    """

    # Load data
    data = pd.read_csv(results_file)  # 'tests\\baseline_tool\\diff_version_timeseries.csv'

    # Ribbon plot with standard deviations
    sns.relplot(
        data=data,
        x='time', y='value',
        hue='version', ci='sd',
        col='version', row='variable',
        kind='line',
    )
    # sns.lineplot(data=data, x='time', y='value', hue='version', ci='sd', row='variable')
    plt.show()

    # Faceted traces for sweep
    sns.relplot(
        data=data,
        x='time', y='value',
        hue='seed', units='seed',
        col='version', row='variable',
        estimator=None, lw=1,
        kind="line",
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--colorblind', help='If included the graph will be colorblind friendly :)', required=False, action='store_true')
    parser.add_argument('-t', '--timeseries', help='Plot the timeseries data', action='store_true')

    args = parser.parse_args()

    if args.timeseries:
        plot_timeseries_comparison()
    else:
        plot_baseline_comparison(args.colorblind)