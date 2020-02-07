import os
from pathlib import Path
import itertools
import argparse
import numpy as np
import pandas as pd
import unidecode
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing

from fp_utils.dhs import DHS
from fp_utils.urhi import URHI
from fp_utils import *

sns.set(font_scale=1.4)
sns.set_style('white')

fs=(12,8)

username = os.path.split(os.path.expanduser('~'))[-1]
folderdict = {
    'dklein': {
        'DHS': '/home/dklein/Dropbox (IDM)/FP Dynamic Modeling/DHS/Country data/Senegal/',
        'URHI': '/home/dklein/Dropbox (IDM)/URHI/Senegal',
    },
    'cliffk': {
        'DHS': '/u/cliffk/idm/fp/data/DHS/NGIR6ADT/NGIR6AFL.DTA',
        'URHI': '/home/cliffk/idm/fp/data/Senegal'
    }
}


def main(show_plots = False, force_read = False, individual_barriers = False):
    results_dir = os.path.join('results', 'Combined')
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    cores = multiprocessing.cpu_count()
    u = URHI(folderdict[username]['URHI'], cores=cores, force_read = force_read)
    d = DHS(folderdict[username]['DHS'], cores=cores, force_read = force_read)

    if individual_barriers:
        ib = d.compute_individual_barriers()
        ib.to_csv(os.path.join(results_dir, 'DHSIndividualBarriers.csv'), na_rep='NaN')
        print(d.data.loc[d.data['MethodType']=='No method'].shape[0])
        print(ib.shape[0])


    # Useful to see which methods are classified as modern / traditional
    print(pd.crosstab(index=d.data['Method'], columns=d.data['MethodType'], values=d.data['Weight']/1e6, aggfunc=sum))
    print(pd.crosstab(index=u.data['Method'], columns=u.data['MethodType'], values=u.data['Weight'], aggfunc=sum))

    cols = ['Survey', 'SurveyName', 'AgeBin', 'AgeBinCoarse', 'Method', 'MethodType', 'MethodDurability', 'Weight', 'Date', 'ParityBin', 'Unmet', 'UID']
    #c = pd.concat((d.dakar_urban[cols], u.data[cols]))
    urhi_like = pd.concat((d.urhi_like[cols], u.data[cols]))
    #c = pd.concat((d.data[cols], u.data[cols]))
    dhs_urhi = pd.concat((d.data[cols], d.urhi_like[cols], u.data[cols]))

    all_data = pd.concat((d.data[cols], d.urhi_like[cols], d.urban[cols], d.rural[cols], u.data[cols]))

    ###########################################################################
    # DHS Barriers
    ###########################################################################


    #print(dhs_urhi.groupby(['Survey', 'SurveyName', 'Date'])['Weight'].sum()/1e6)
    #exit()

    ib = d.get_individual_barriers()

    stacked = ib.stack()
    #stacked.name = 'Stack'
    stacked.index.rename('Barrier', level=8, inplace=True)
    stacked.name = 'Count'
    stacked = stacked.reset_index('Weight')

    stacked.loc[:,'WeightCount'] = stacked['Weight'] * stacked['Count'] / 1e6

    stacked.reset_index(inplace=True)

    stacked_barriers_weight_sum = stacked.groupby(['SurveyName', 'Barrier'])['WeightCount'].sum()
    no_method_weight_sum = d.data.loc[d.data['Method']=='No method'].groupby('SurveyName')['Weight'].sum() / 1e6
    weight_sum = d.data.groupby('SurveyName')['Weight'].sum() / 1e6

    fig, ax = plt.subplots(1,1,figsize=fs)
    percent_on_no_method = 100 *no_method_weight_sum.divide(weight_sum)
    percent_on_no_method.plot(ax=ax)
    ax.set_xlabel('Survey')
    ax.set_ylabel('Women Not Using Any Method (%)')
    #ax.set_ylim(bottom=0, top=100)
    fig.savefig(os.path.join(results_dir, 'NonUse.png'))


    barrier_percent_of_no_need = 100 * stacked_barriers_weight_sum.divide(no_method_weight_sum)

    #bar_data = stacked.groupby(['SurveyName', 'Barrier'])['WeightCount'].sum()
    print(barrier_percent_of_no_need)
    print(barrier_percent_of_no_need.groupby('SurveyName').sum())

    unstack_for_total = barrier_percent_of_no_need.unstack('Barrier')
    unstack_for_total['N/A'] = unstack_for_total['N/A'] + 100 - unstack_for_total.sum(axis=1)
    unstack_for_total.rename(columns={'N/A':'Missing'}, inplace=True)
    unstack_for_total.name = 'Percent'
    barrier_percent_of_no_need_with_missing = unstack_for_total.stack()

    #stacked.loc[:,'WeightCount'] = stacked['Weight'] * stacked['Count']
    #bar = stacked.groupby(['SurveyName', 'Barrier'])['WeightCount'].sum()
    fig, ax = plt.subplots(1,1,figsize=fs)
    barrier_percent_of_no_need_with_missing.unstack('SurveyName').plot.bar(rot=0, ax=ax)
    ax.set_ylabel('Barriers reported (percent of non users)')
    fig.savefig(os.path.join(results_dir, 'Barriers.png'))


    ###########################################################################
    # URHI - who converted to implants?
    ###########################################################################
    implant_end = u.data.loc[(u.data['Method']=='Implant') & (u.data['SurveyName']=='Endline')]
    switchers = u.data.loc[u.data['SurveyName']=='Baseline'] \
        .set_index('UID') \
        .loc[implant_end['UID']]
    switchers = switchers.loc[switchers['Method'] != 'Implant']

    switchers_ct_raw = pd.crosstab(index=switchers['AgeBin'], columns=switchers['ParityBin'], values=switchers['Weight'], aggfunc=sum)
    ct_distribution = 100 * switchers_ct_raw / switchers_ct_raw.sum().sum()
    ct_distribution.fillna(0, inplace=True)
    ct_distribution.to_csv(os.path.join(results_dir, 'URHI_SwitchedToImplants_FromAgeParity_Distribution.csv'))

    switch_from = 100 * switchers.groupby('Method')['Weight'].sum() .divide( switchers['Weight'].sum() ).fillna(0)

    endline = u.data.loc[u.data['SurveyName']=='Endline']
    total = u.data.loc[u.data['SurveyName']=='Baseline'] \
        .set_index('UID') \
        .loc[endline['UID']]
    total = total.loc[total['Method'] != 'Implant']

    ct_total = pd.crosstab(index=total['AgeBin'], columns=total['ParityBin'], values=total['Weight'], aggfunc=sum)

    switch_frac = 100*switchers_ct_raw.divide(ct_total).fillna(0)
    switch_parity_frac = 100 * switchers_ct_raw.sum(axis=0).divide(ct_total.sum(axis=0)).fillna(0)
    switch_age_frac = 100 * switchers_ct_raw.sum(axis=1).divide(ct_total.sum(axis=1)).fillna(0)

    num_by_method = switchers.groupby('Method')['Weight'].sum()
    den_by_method =total.groupby('Method')['Weight'].sum()

    switch_from_popfrac = 100 * num_by_method.divide(den_by_method).fillna(0)


    switchers_ct_raw.to_csv(os.path.join(results_dir, 'URHI_SwitchedToImplants_FromAgeParity_Counts.csv'))
    ct_total.to_csv(os.path.join(results_dir, 'URHI_SwitchedToImplants_FromAgeParity_PopTotals.csv'))
    switch_frac.to_csv(os.path.join(results_dir, 'URHI_SwitchedToImplants_FromAgeParity_PopFraction.csv'))
    switch_parity_frac.to_csv(os.path.join(results_dir, 'URHI_SwitchedToImplants_FromParity_PopFraction.csv'))
    switch_age_frac.to_csv(os.path.join(results_dir, 'URHI_SwitchedToImplants_FromAge_PopFraction.csv'))

    switch_from.to_csv(os.path.join(results_dir, 'URHI_SwitchedToImplants_FromMethods_Distribution.csv'))
    switch_from_popfrac.to_csv(os.path.join(results_dir, 'URHI_SwitchedToImplants_FromMethods_PopFrac.csv'))

    with pd.option_context('display.max_rows', 1000, 'display.float_format', '{:.1f}'.format): # 'display.precision',2, 
        print('Switchers CT:\n', switchers_ct_raw)
        print('Switchers CT distribution\n:', ct_distribution)
        print('Total CT:\n', ct_total)
        print('Switchers fraction of Total:\n', switch_parity_frac)
        print( switch_parity_frac )
        print( switch_age_frac )

        print('Switched to implants from [Distribution]:\n', switch_from)
        print('Switched to implants from [Pop Frac]:\n', switch_from_popfrac)


    dat2011 = pd.concat( [
        #d.data[cols].loc[d.data['SurveyName']=='2010-11'],
        d.urhi_like[cols].loc[d.urhi_like['SurveyName']=='2010-11'], # URHI-like
        #d.dakar_urban[cols].loc[d.dakar_urban['SurveyName']=='2010-11'], # Dakar-urban
        u.data[cols].loc[u.data['SurveyName']=='Baseline']
    ])
    dat2011['Year'] = 2011

    ### Stacked bar code is SUPER UGLY - TODO
    fig, ax = plt.subplots(1,1,figsize=(12,6))
    tmp = dat2011.groupby(['Survey', 'MethodType', 'AgeBin'])['Weight'].sum()
    wsum = dat2011.groupby(['Survey'])['Weight'].sum()
    tmp = tmp.divide(wsum).reset_index()
    tmp.loc[tmp['Weight'].isna(), 'Weight'] = 0
    X = np.arange( tmp['AgeBin'].nunique() )
    width = 0.35
    col = ['r', 'g', 'b', 'm', 'y', 'c']
    hatch = [None, '/']
    for survey_index, (survey, dat_survey) in enumerate(tmp.groupby('Survey')):
        bot = None
        for method_idx, (method, dat_method) in enumerate(dat_survey.groupby('MethodType')):
            label=None
            if survey_index == 0:
                label=method
            plt.bar(X+survey_index*width, height=100*dat_method['Weight'], width=width, bottom=bot, label=label, color=col[method_idx], hatch=hatch[survey_index])

            if isinstance(bot, np.ndarray):
                bot = bot + 100*dat_method['Weight'].values
            else:
                bot = 100*dat_method['Weight'].values
    ax.set_xticks(X)
    ax.set_xticklabels(tmp['AgeBin'].unique())
    ax.set_xlabel('Age')
    ax.set_ylabel('Percent')
    plt.legend()
    fig.savefig(os.path.join(results_dir, 'MethodTypeBar_by_AgeBin.png'))

    ### Stacked bar code is SUPER UGLY - TODO
    fig, ax = plt.subplots(1,1,figsize=(12,6))
    tmp = dat2011.groupby(['Survey', 'MethodType', 'ParityBin'])['Weight'].sum()
    wsum = dat2011.groupby(['Survey'])['Weight'].sum()
    tmp = tmp.divide(wsum).reset_index()
    tmp.loc[tmp['Weight'].isna(), 'Weight'] = 0
    X = np.arange( tmp['ParityBin'].nunique() )
    width = 0.35
    col = ['r', 'g', 'b', 'm', 'y', 'c']
    hatch = [None, '/']
    for survey_index, (survey, dat_survey) in enumerate(tmp.groupby('Survey')):
        bot = None
        for method_idx, (method, dat_method) in enumerate(dat_survey.groupby('MethodType')):
            label=None
            if survey_index == 0:
                label=method
            plt.bar(X+survey_index*width, height=100*dat_method['Weight'], width=width, bottom=bot, label=label, color=col[method_idx], hatch=hatch[survey_index])

            if isinstance(bot, np.ndarray):
                bot = bot + 100*dat_method['Weight'].values
            else:
                bot = 100*dat_method['Weight'].values
    ax.set_xticks(X)
    ax.set_xticklabels(tmp['ParityBin'].unique())
    ax.set_xlabel('Parity')
    ax.set_ylabel('Percent')
    plt.legend()
    fig.savefig(os.path.join(results_dir, 'MethodTypeBar_by_ParityBin.png'))


    ###########################################################################
    # SURVEY SIZE
    ###########################################################################
    print(dhs_urhi.groupby(['Survey', 'SurveyName', 'Date']).size())
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    tmp = pd.concat((d.data[cols], d.urhi_like[cols], d.urban[cols], u.data[cols])) \
        .groupby(['Survey', 'Date']).size()
    tmp.name = 'Sample Size'
    sns.lineplot(data = tmp.reset_index(), x='Date', y='Sample Size', hue='Survey', marker='o', markersize=10)
    plt.ylim((0,None))
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'SurveySize.png'))

    ###########################################################################
    # MCPR - All women only for now
    ###########################################################################
    # TODO: pull out married women, or exposed
    tmp = all_data.copy()
    tmp['Modern'] = tmp['MethodType'] == 'Modern'

    g = sns.FacetGrid(data=tmp, hue='Survey', height=8, legend_out=False, aspect=0.75)
    g.map_dataframe(plot_line_percent, by='Modern', values=[True]).add_legend().set_xlabels('Year').set_ylabels('mCPR-All Women').set(ylim=(0,None))
    g.savefig(os.path.join(results_dir, 'mCPR.png'))

    ###########################################################################
    # POPULATION PYRAMID BY AGE
    ###########################################################################
    '''
    g = sns.FacetGrid(data=urhi_like, hue='SurveyName', height=5)
    g.map_dataframe(plot_pop_pyramid).add_legend().set_xlabels('Percent').set_ylabels('Age Bin')
    g.savefig(os.path.join(results_dir, 'PopulationPyramid.png'))
    '''

    ###########################################################################
    # UNMET NEED
    ###########################################################################
    g = sns.FacetGrid(data=urhi_like, height=5)
    g.map_dataframe(plot_line_percent, by='Unmet').add_legend().set_xlabels('Year').set_ylabels('Percent').set(ylim=(0,None))
    g.savefig(os.path.join(results_dir, 'UnmetNeed.png'))

    # Move Unknown to No, only affects URHI.  Should compute my own Unmet Need column
    #print(u.data.loc[u.data['Unmet']=='Unknown', ['Method']])
    tmp = dhs_urhi.loc[dhs_urhi['Date']>2005,:]
    tmp = tmp.loc[tmp['Unmet'].isin(['Yes', 'No'])]
    g = sns.FacetGrid(data=tmp, hue='Survey', height=8, legend_out=False, aspect=7/5)
    g.map_dataframe(plot_line_percent, by='Unmet', values=['Yes']).add_legend().set_xlabels('Year').set_ylabels('Unmet Need').set(ylim=(0,None))
    g.savefig(os.path.join(results_dir, 'Unmet_UnknownAsNo.png'))


    ###########################################################################
    # METHOD TYPE LINES
    ###########################################################################
    g = sns.FacetGrid(data=urhi_like, height=5)
    g.map_dataframe(plot_line_percent, by='MethodType').add_legend().set_xlabels('Year').set_ylabels('Percent').set(ylim=(0,None))
    g.savefig(os.path.join(results_dir, 'MethodType.png'))

    g = sns.FacetGrid(data=urhi_like, col='AgeBinCoarse', height=5)
    g.map_dataframe(plot_line_percent, by='MethodType').add_legend().set_xlabels('Year').set_ylabels('Percent').set(ylim=(0,None))
    g.savefig(os.path.join(results_dir, 'MethodType_by_AgeBinCoarse.png'))

    g = sns.FacetGrid(data=urhi_like, col='ParityBin', col_wrap=4, height=3)
    g.map_dataframe(plot_line_percent, by='MethodType').add_legend().set_xlabels('Year').set_ylabels('Percent').set(ylim=(0,None))
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
        #d.data[cols].loc[d.data['SurveyName']=='2015'],
        d.urhi_like[cols].loc[d.urhi_like['SurveyName']=='2015'], # URHI-like
        #d.dakar_urban[cols].loc[d.dakar_urban['SurveyName']=='2010-11'], # Dakar-urban
        u.data[cols].loc[u.data['SurveyName']=='Endline']
    ])
    dat2015['Year'] = 2015

    dat = dat2011 #pd.concat([dat2011, dat2015])
    mod = dat.loc[dat['MethodType']=='Modern']

    with plt.style.context({"axes.prop_cycle" : plt.cycler("color", plt.cm.tab20.colors)}):
        g = sns.FacetGrid(data=dat, col='Year', row='Survey', height=8, aspect=2)
        g.map_dataframe(plot_pie, by='MethodType')#.add_legend()#.set_xlabels('Year').set_ylabels('Percent')
        g.savefig(os.path.join(results_dir, 'MethodPies_by_Survey_Year.png'))

        '''
        g = sns.FacetGrid(data=dat, col='Year', row='Survey', height=8)
        g.map_dataframe(plot_pie, by='MethodDurability')#.add_legend()#.set_xlabels('Year').set_ylabels('Percent')
        g.savefig(os.path.join(results_dir, 'MethodPies_by_Durability.png'))

        g = sns.FacetGrid(data=mod, col='Year', row='Survey', height=8)
        g.map_dataframe(plot_pie, by='MethodDurability')#.add_legend()#.set_xlabels('Year').set_ylabels('Percent')
        g.savefig(os.path.join(results_dir, 'NotNoneMethodPies_by_Durability.png'))
        '''

    with plt.style.context({"axes.prop_cycle" : plt.cycler("color", plt.cm.Set1.colors)}):
        g = sns.FacetGrid(data=mod, col='Year', row='Survey', height=8, aspect=2)
        g.map_dataframe(plot_pie, by='Method')#.add_legend()#.set_xlabels('Year').set_ylabels('Percent')
        g.savefig(os.path.join(results_dir, 'ModernMethodPies_by_Survey_Year.png'))


    ###########################################################################
    # MODERN METHOD LINES
    ###########################################################################
    data = urhi_like
    mod = data.loc[(data['MethodType']=='Modern') & (data['Date']>1995)]

    g = sns.FacetGrid(data=mod, height=5)
    g.map_dataframe(plot_line_percent, by='Method').add_legend().set_xlabels('Year').set_ylabels('Percent').set(ylim=(0,None))
    g.savefig(os.path.join(results_dir, 'ModernMethod.png'))

    g = sns.FacetGrid(data=mod, col='AgeBinCoarse', height=5)
    g.map_dataframe(plot_line_percent, by='Method').add_legend().set_xlabels('Year').set_ylabels('Percent').set(ylim=(0,None))
    g.savefig(os.path.join(results_dir, 'ModernMethod_by_AgeBinCoarse.png'))

    g = sns.FacetGrid(data=mod, col='ParityBin', col_wrap=4, height=3)
    g.map_dataframe(plot_line_percent, by='Method').add_legend().set_xlabels('Year').set_ylabels('Percent').set(ylim=(0,None))
    g.savefig(os.path.join(results_dir, 'ModernMethod_by_Parity.png'))

    mod.loc[mod['Method'].isin(['Female sterilization', 'LAM', 'IUD', 'Condom']), 'Method'] = 'Other modern'
    g = sns.FacetGrid(data=mod, height=8)
    g.map_dataframe(plot_line_percent, by='Method', hue_order=['Injectable', 'Daily pill', 'Other modern', 'Implant']).add_legend().set_xlabels('Year').set_ylabels('Percent').set(ylim=(0,None))
    g.savefig(os.path.join(results_dir, 'ModernMethodCoarse.png'))

    ###########################################################################
    # METHODTYPE STACK
    ###########################################################################
    g = sns.FacetGrid(data=urhi_like, col='Survey', height=10, aspect=0.8, sharex=False, legend_out=False)
    g.map_dataframe(plot_stack, by='MethodType', order=['Modern', 'Traditional', 'No method']).add_legend().set_xlabels('Year').set_ylabels('Percent') \
        .set(xlim=(d.data['Date'].min(), d.data['Date'].max()))
    plt.subplots_adjust(right=0.9)
    g.savefig(os.path.join(results_dir, 'MethodTypeStack.png'))

    #Using - want vs Not using by want to vs Not using  do not want to - add to 100%
    data = urhi_like.copy()
    data.loc[data['MethodType']=='Modern','Use'] = 'Modern'
    data.loc[data['MethodType']=='Traditional','Use'] = 'Traditional'
    data.loc[ (data['MethodType']=='No method') & (data['Unmet']=='Yes'),'Use'] = 'None-Unmet'
    data.loc[ (data['MethodType']=='No method') & (data['Unmet']=='Unknown'),'Use'] = 'None-Unknown'
    data.loc[ (data['MethodType']=='No method') & (data['Unmet']=='No'),'Use'] = 'None-Met'

    g = sns.FacetGrid(data=data, col='Survey', height=10, aspect=0.8, sharex=False, legend_out=False)
    g.map_dataframe(plot_stack, by='Use', order=['Modern', 'Traditional', 'None-Unmet', 'None-Met', 'None-Unknown']).add_legend().set_xlabels('Year').set_ylabels('Percent') \
        .set(xlim=(data['Date'].min(), data['Date'].max()))
    plt.subplots_adjust(right=0.9)
    g.savefig(os.path.join(results_dir, 'MethodUseStack.png'))


    data.loc[data['MethodType']=='Modern','Use'] = data.loc[data['MethodType']=='Modern','Method']
    data.loc[data['Use'].isin(['Female sterilization', 'LAM', 'IUD', 'Condom']), 'Use'] = 'Other modern'
    data = data.loc[data['Date'] > 2005]
    data = data.loc[data['Survey'] == 'URHI']
    g = sns.FacetGrid(data=data, col='Survey', height=10, aspect=0.8, sharex=False)
    g.map_dataframe(plot_stack, by='Use', order=['Daily pill', 'Other modern', 'Injectable', 'Implant', 'Traditional', 'None-Unmet', 'None-Met', 'None-Unknown']).add_legend().set_xlabels('Year').set_ylabels('Percent') \
        .set(xlim=(data['Date'].min(), data['Date'].max()))
    plt.subplots_adjust(right=0.9)
    g.savefig(os.path.join(results_dir, 'MethodUseStackWithModern.png'))



    ###########################################################################
    # SKYSCRAPER IMAGES
    ###########################################################################
    g = sns.FacetGrid(data=dat, col='Survey', row='Year', height=5)
    g.map_dataframe(plot_skyscraper, age='AgeBin', parity='ParityBin', vmax=20).set_xlabels('Parity').set_ylabels('Age')
    g.savefig(os.path.join(results_dir, 'Skyscraper.png'))

    dat_unmet = dat.loc[dat['Unmet']=='Yes']
    g = sns.FacetGrid(data=dat_unmet, col='Survey', row='Year', height=5)
    g.map_dataframe(plot_skyscraper, age='AgeBin', parity='ParityBin', vmax=20).set_xlabels('Parity').set_ylabels('Age')
    g.savefig(os.path.join(results_dir, 'SkyscraperUnmet.png'))



    # CURRENTLY PREGNANT - don't have from URHI yet
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

    if show_plots:
        plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--barriers', default=False, action='store_true')
    parser.add_argument('--plot', default=False, action='store_true')
    args = parser.parse_args()

    main(show_plots = args.plot, force_read = args.force, individual_barriers = args.barriers)


#print(pd.crosstab(data['SurveyName'], data['v102'], data['v213']*data['Weight']/1e6, aggfunc=sum))
#with pd.option_context('display.precision', 1, 'display.max_rows', 1000): # 'display.precision',2, 
