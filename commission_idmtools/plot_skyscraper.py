import os
import pandas as pd
import argparse
from analyze import analyze

def plot(exp_id):
    data = pd.read_csv(os.path.join(exp_id, 'CSV_Analyzer.csv'), skipinitialspace=True)
    tags = pd.read_csv(os.path.join(exp_id, 'Tags_Analyzer.csv'), skipinitialspace=True)

    print(data)
    print(tags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', '-I', action='store', dest='exp_id', required=True)
    parser.add_argument('--platform', '-P', action='store', dest='platform', default='COMPS')
    parse = parser.parse_args()

    if not os.path.isdir(parse.exp_id):
        analyze(parse.exp_id, parse.platform)

    plot(parse.exp_id)
