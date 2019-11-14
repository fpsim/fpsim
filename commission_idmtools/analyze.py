from idmtools.analysis.analyze_manager import AnalyzeManager
from idmtools.analysis.download_analyzer import DownloadAnalyzer
from idmtools.core import ItemType

from idmtools.core.platform_factory import Platform
from idmtools.entities import IAnalyzer

import os, sys
import pandas as pd
import argparse


class CSV_Analyzer(IAnalyzer):
    def __init__(self, filenames, uid=None, working_dir=None, parse=True):
        super().__init__(uid, working_dir, parse, filenames=filenames)

    def filter(self, simulation: 'TSimulation') -> bool:
        #return int(simulation.tags.get("b")) > 5
        # NOTE: Only succeded simulations make it here, see:
        # https://github.com/InstituteforDiseaseModeling/idmtools/issues/457
        return True

    def map(self, data, simulation):
        return data[self.filenames[0]]

    def reduce(self, all_data):
        # Let's hope the first simulation is representative
        first_sim = next(iter(all_data.keys()))
        exp_id = str(first_sim.experiment.uid)

        results = pd.concat(list(all_data.values()), axis=0,
            keys = [str(k.uid) for k in all_data.keys()],
            names = ['SimId'])
        results.index = results.index.droplevel(1) # Remove default index

        os.makedirs(exp_id, exist_ok=True)
        results.to_csv( os.path.join( exp_id, self.__class__.__name__+'.csv') )


class Tags_Analyzer(IAnalyzer):
    def __init__(self, uid=None, working_dir=None, parse=True):
        super().__init__(uid, working_dir, parse)
        self.exp_id = None

    def filter(self, simulation: 'TSimulation') -> bool:
        return True

    def map(self, data, simulation):
        df = pd.DataFrame(columns = simulation.tags.keys())
        df.loc[str(simulation.uid)] = list(simulation.tags.values())
        df.index.name = 'SimId'
        return df

    def reduce(self, all_data):
        exp_id = str(next(iter(all_data.keys())).experiment.uid)
        results = pd.concat(list(all_data.values()), axis=0)
        os.makedirs(exp_id, exist_ok=True)
        results.to_csv( os.path.join( exp_id, self.__class__.__name__+'.csv') )


def analyze():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', '-I', action='store', dest='exp_id')
    parser.add_argument('--platform', '-P', action='store', dest='platform', default='COMPS')
    parse = parser.parse_args()

    print(f'Analyzing experiment on {parse.platform} with id = {parse.exp_id}')

    platform = Platform(parse.platform)

    ReportFPBAP = CSV_Analyzer(filenames = ["output/ReportFPByAgeAndParity.csv"])
    Tags = Tags_Analyzer()
    '''
    ReportFPBAP_anlayzer = DownloadAnalyzer(
        filenames=['output/ReportFPByAgeAndParity.csv'],
        output_path = parse.exp_id
    )
    '''

    experiment_tuple = (parse.exp_id, ItemType.EXPERIMENT)

    manager = AnalyzeManager(platform=platform, ids=[experiment_tuple])
    manager.add_analyzer(Tags)
    manager.add_analyzer(ReportFPBAP)
    manager.analyze()

if __name__ == "__main__":
    analyze()
