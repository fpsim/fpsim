import pandas as pd
import matplotlib.pyplot as plt

do_plot = 1
# Data files must be raw individual recode DHS data (available for request at dhsprogram.com), in STATA format (.dta).
# For example, Kenya's 2014 DHS data would be 'KEIR72FL.DTA'
data_file = 'Your file here'

# Read in DHS data
dhs = pd.read_stata(data_file, convert_categoricals = False)

# To truncate parity at 7+
parity_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:7, 9:7, 10:7, 11:7, 12:7, 13:7}
dhs['v201'] = dhs['v201'].map(parity_dict)


def parity_mcpr(data, name, parity, method, weight):
    total = data.groupby(parity)[weight].sum()
    parity_group = data.groupby([parity, method])[weight].sum()
    if data is dhs:
        n = total / 1000000
    else:
        n = total
    totals = n
    parity_group = 100 * parity_group.divide(total).fillna(0)
    parity_group.name = 'Percent'
    return parity_group, totals

def caculate_mcpr(data):
    method_group = data.groupby(['v313'])['v005'].sum()
    total = data['v005'].sum()
    method_group = 100 * method_group.divide(total).fillna(0)
    method_group.name = 'Percent'
    return method_group[3].round(1)

def process_mcpr(data, method):
    data.reset_index(inplace = True)
    modern = data[data[method]==3]
    modern.reset_index(drop=True, inplace=True)
    return modern

def process_tcpr(data, method):
    data.reset_index(inplace = True)
    trad = data[data[method]==2]
    trad.reset_index(drop=True, inplace=True)
    return trad

mcpr_overall = caculate_mcpr(data=dhs) # Calculate total mCPR in DHS data to check matches STAT Compiler.  Target is 39.1% for Kenya.
print(f'mCPR calculated for DHS:  {mcpr_overall}')
table_dhs, totals = parity_mcpr(data=dhs, name='percent', parity='v201', method='v313', weight='v005')

# Create DataFrames and select out only modern or traditional contraception
parity_df_mod_dhs = pd.DataFrame(table_dhs).copy(deep=True)
parity_df_trad_dhs = pd.DataFrame(table_dhs).copy(deep=True)

parity_modern_dhs = process_mcpr(parity_df_mod_dhs, method='v313')
parity_trad_dhs = process_tcpr(parity_df_trad_dhs, method='v313')

parity_modern_dhs['totals'] = totals
overall = parity_modern_dhs['totals'].sum()
overall_percent = (parity_modern_dhs['totals'] / overall) * 100
parity_modern_dhs['parity percent'] = overall_percent
parity_modern_dhs = parity_modern_dhs.rename(columns={'v201': 'parity', 'v313': 'method', 'Percent': 'mcpr'})
print(parity_modern_dhs)


# Plotting
if do_plot:
    mCPR_dhs = parity_modern_dhs['mcpr']
    ind = parity_modern_dhs['parity']
    width = 0.35

    fig = plt.figure(figsize=(9, 7), dpi=120)
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, mCPR_dhs, width, color='royalblue', label = f'Data - overall mCPR {mcpr_overall}')

    ax.set_ylabel('mCPR - %')
    ax.set_xlabel('Parity')
    ax.set_title('mCPR by parity in DHS data') # change title to reflect DHS country and year being used
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels( ('0', '1', '2', '3', '4', '5', '6', '7+'))

    ax.legend(loc = 'upper left')

    plt.savefig('Insert file name here') #eg: kenya_mcpr_parity.png
















