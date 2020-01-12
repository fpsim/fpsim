import os
from pathlib import Path
import itertools
import argparse
import numpy as np
import pandas as pd
import unidecode
import seaborn as sns
import matplotlib.pyplot as plt

from fp_utils.dhs import DHS
from fp_utils.urhi import URHI
from fp_utils import plot_line, plot_pie, plot_pop_pyramid, plot_skyscraper

fs=(12,8)

username = os.path.split(os.path.expanduser('~'))[-1]
folderdict = {
    'dklein': {
        'DHS': '/home/dklein/sdb2/Dropbox (IDM)/FP Dynamic Modeling/DHS/Country data/Senegal/',
        'URHI': '/home/dklein/sdb2/Dropbox (IDM)/URHI/Senegal',
    },
    'cliffk': {
        'DHS': '/u/cliffk/idm/fp/data/DHS/NGIR6ADT/NGIR6AFL.DTA',
        'URHI': '/home/cliffk/idm/fp/data/Senegal'
    }
}


def main(force_read = False):
    results_dir = os.path.join('results', 'Combined')
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    d = DHS(folderdict[username]['DHS'], force_read)
    u = URHI(folderdict[username]['URHI'], force_read)

    # Useful to see which methods are classified as modern / traditional
    print(pd.crosstab(index=d.data['Method'], columns=d.data['MethodType'], values=d.data['Weight']/1e6, aggfunc=sum))
    print(pd.crosstab(index=u.data['Method'], columns=u.data['MethodType'], values=u.data['Weight'], aggfunc=sum))

    cols = ['Survey', 'SurveyName', 'AgeBin', 'AgeBinCoarse', 'Method', 'MethodType', 'Weight', 'Date', 'ParityBin', 'Unmet', 'UID']
    #c = pd.concat((d.dakar_urban[cols], u.data[cols]))
    c = pd.concat((d.urhi_like[cols], u.data[cols]))
    #c = pd.concat((d.data[cols], u.data[cols]))


    # Age pyramid
    g = sns.FacetGrid(data=c, hue='SurveyName', height=5)
    g.map_dataframe(plot_pop_pyramid).add_legend().set_xlabels('Percent').set_ylabels('Age Bin')
    g.savefig(os.path.join(u.results_dir, 'PopulationPyramid.png'))


    ###########################################################################
    # UNMET NEED
    ###########################################################################
    g = sns.FacetGrid(data=c.loc[(c['Date']>1990) & (c['Unmet']!='Unknown')], height=5)
    g.map_dataframe(plot_line, by='Unmet').add_legend().set_xlabels('Year').set_ylabels('Percent')
    g.savefig(os.path.join(results_dir, 'UnmetNeed.png'))


    ###########################################################################
    # METHOD TYPE LINES
    ###########################################################################
    g = sns.FacetGrid(data=c, height=5)
    g.map_dataframe(plot_line, by='MethodType').add_legend().set_xlabels('Year').set_ylabels('Percent')
    g.savefig(os.path.join(results_dir, 'MethodType.png'))

    g = sns.FacetGrid(data=c, col='AgeBinCoarse', height=5)
    g.map_dataframe(plot_line, by='MethodType').add_legend().set_xlabels('Year').set_ylabels('Percent')
    g.savefig(os.path.join(results_dir, 'MethodType_by_AgeBinCoarse.png'))

    g = sns.FacetGrid(data=c, col='ParityBin', col_wrap=4, height=3)
    g.map_dataframe(plot_line, by='MethodType').add_legend().set_xlabels('Year').set_ylabels('Percent')
    g.savefig(os.path.join(results_dir, 'MethodType_by_ParityBin.png'))


    ###########################################################################
    # METHOD PIES
    ###########################################################################
    dat2011 = pd.concat( [
        #d.data[cols].loc[d.data['SurveyName']=='2010-11'],
        d.urhi_like[cols].loc[d.urhi_like['SurveyName']=='2010-11'], # URHI-like
        #d.dakar_urban[cols].loc[d.dakar_urban['SurveyName']=='2010-11'], # Dakar-urban
        u.data[cols].loc[u.data['SurveyName']=='Baseline']
    ])
    dat2011['Year'] = 2011

    dat2015 = pd.concat( [
        #d.data[cols].loc[d.data['SurveyName']=='2010-11'],
        d.urhi_like[cols].loc[d.urhi_like['SurveyName']=='2015'], # URHI-like
        #d.dakar_urban[cols].loc[d.dakar_urban['SurveyName']=='2010-11'], # Dakar-urban
        u.data[cols].loc[u.data['SurveyName']=='Endline']
    ])
    dat2015['Year'] = 2015

    dat = pd.concat([dat2011, dat2015])

    g = sns.FacetGrid(data=dat, col='Survey', row='Year', height=5)
    g.map_dataframe(plot_pie, by='MethodType').add_legend()#.set_xlabels('Year').set_ylabels('Percent')
    g.savefig(os.path.join(results_dir, 'MethodPies_by_Survey_Year.png'))

    mod = dat.loc[dat['MethodType']=='Modern']
    g = sns.FacetGrid(data=mod, col='Survey', row='Year', height=5)
    g.map_dataframe(plot_pie, by='Method').add_legend()#.set_xlabels('Year').set_ylabels('Percent')
    g.savefig(os.path.join(results_dir, 'ModernMethodPies_by_Survey_Year.png'))

    # Skyscraper images
    g = sns.FacetGrid(data=dat, col='Survey', row='Year', height=5)
    g.map_dataframe(plot_skyscraper, age='AgeBin', parity='ParityBin', vmax=20).set_xlabels('Parity').set_ylabels('Age')
    g.savefig(os.path.join(u.results_dir, 'Skyscraper.png'))



    ###########################################################################
    # MODERN METHOD LINES
    ###########################################################################

    mod = c.loc[(c['MethodType']=='Modern') & (c['Date']>1990)]

    g = sns.FacetGrid(data=mod, height=5)
    g.map_dataframe(plot_line, by='Method').add_legend().set_xlabels('Year').set_ylabels('Percent')
    g.savefig(os.path.join(results_dir, 'ModernMethod.png'))

    g = sns.FacetGrid(data=mod, col='AgeBinCoarse', height=5)
    g.map_dataframe(plot_line, by='Method').add_legend().set_xlabels('Year').set_ylabels('Percent')
    g.savefig(os.path.join(results_dir, 'ModernMethod_by_AgeBinCoarse.png'))

    g = sns.FacetGrid(data=mod, col='ParityBin', col_wrap=4, height=3)
    g.map_dataframe(plot_line, by='Method').add_legend().set_xlabels('Year').set_ylabels('Percent')
    g.savefig(os.path.join(results_dir, 'ModernMethod_by_Parity.png'))


    ###########################################################################
    # URHI - who converted to implants?
    ###########################################################################
    implant_end = u.data.loc[(u.data['Method']=='Implant') & (u.data['SurveyName']=='Endline')]
    switchers = u.data.loc[u.data['SurveyName']=='Baseline'] \
        .set_index('UID') \
        .loc[implant_end['UID']]
    switchers = switchers.loc[switchers['Method'] != 'Implant']

    ct = pd.crosstab(index=switchers['AgeBin'], columns=switchers['ParityBin'], values=switchers['Weight'], aggfunc=sum)
    ct = 100 * ct / ct.sum().sum()
    ct.fillna(0, inplace=True)
    ct.to_csv(os.path.join(results_dir, 'URHI_EndImplants_FromAgeParity.csv'))

    switch_from = 100 * switchers.groupby('Method')['Weight'].sum() / switchers['Weight'].sum()
    switch_from.to_csv(os.path.join(results_dir, 'URHI_EndImplants_FromMethods.csv'))

    with pd.option_context('display.max_rows', 1000, 'display.float_format', '{:.0f}'.format): # 'display.precision',2, 
        print(ct)
        print('Switched to implants from:')
        print(switch_from)


    plt.show()

    exit()



    # CURRENTLY PREGNANT
    '''
    fig, ax = plt.subplots(1,1, figsize=fs)
    boolean_plot('Currently pregnant', 'v213', ax=ax[0])
    plt.tight_layout()

    boolean_plot_by('Currently pregnant', 'v213', 'v101')
    boolean_plot_by('Currently pregnant', 'v213', 'v102')

    multi_plot('Unmet need', 'v624')

    fig, ax = plt.subplots(1,4, sharey=True, figsize=fs)
    multi_plot('Method type', 'v313', ax=ax[0])
    plt.tight_layout()
    '''

    multi_plot('ByMethod', 'Method')
    plt.tight_layout()

    plot_pie('All women', data)
    plt.savefig(os.path.join(results_dir, f'Pie-All.png'))

    tmp = data.loc[data['MethodType']=='Modern']
    tmp['Method'] = tmp.Method.cat.remove_unused_categories()
    plot_pie('Modern', tmp)
    plt.savefig(os.path.join(results_dir, f'Pie-Modern.png'))

    # Age pyramids
    age_pyramid_plot('Population Pyramid - URHI', data)

    # Skyscraper images
    skyscraper(data.loc[data['SurveyName']!='Midline'], 'URHI')

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', default=False, action='store_true')
    args = parser.parse_args()

    main(force_read = args.force)


#print(pd.crosstab(data['SurveyName'], data['v102'], data['v213']*data['Weight']/1e6, aggfunc=sum))
#with pd.option_context('display.precision', 1, 'display.max_rows', 1000): # 'display.precision',2, 
