from idmtools.analysis.analyze_manager import AnalyzeManager
from idmtools.analysis.download_analyzer import DownloadAnalyzer
from idmtools.analysis.tags_analyzer import TagsAnalyzer
from idmtools.analysis.csv_analyzer import CSVAnalyzer
from idmtools.core import ItemType

from idmtools.core.platform_factory import Platform
from idmtools.entities import IAnalyzer

import os, sys
import pandas as pd
import argparse


def analyze(exp_id, platform):

    print(f'Analyzing experiment on {platform} with id = {exp_id}')

    platform = Platform(platform)

    ReportFPBAP = CSVAnalyzer(filenames = ["output/ReportFPByAgeAndParity.csv"])
    tags = TagsAnalyzer()
    '''
    ReportFPBAP_anlayzer = DownloadAnalyzer(
        filenames=['output/ReportFPByAgeAndParity.csv'],
        output_path = exp_id
    )
    '''

    experiment_tuple = (exp_id, ItemType.EXPERIMENT)

    manager = AnalyzeManager(platform=platform, ids=[experiment_tuple])
    manager.add_analyzer(tags)
    manager.add_analyzer(ReportFPBAP)
    manager.analyze()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', '-I', action='store', dest='exp_id', required=True)
    parser.add_argument('--platform', '-P', action='store', dest='platform', default='COMPS')
    parse = parser.parse_args()

    analyze(parse.exp_id, parse.platform)
