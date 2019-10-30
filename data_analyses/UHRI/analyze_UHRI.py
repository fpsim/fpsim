# https://www.onlinedoctranslator.com/en/translationprocess
import pandas as pd
from googletrans import Translator

translator = Translator()

filenames = [
    '/home/dklein/Dropbox (IDM)/URHI/Senegal/Baseline/SEN_base_wm_20160427.dta',
    '/home/dklein/Dropbox (IDM)/URHI/Senegal/Midline/SEN_mid_wm_match_20160609.dta',
    '/home/dklein/Dropbox (IDM)/URHI/Senegal/Endline/SEN_end_wm_match_20160505.dta'
]

data = pd.read_stata(filename)

print(data.iloc[0])

def translate(data):
    if data[0].isdigit() and ' ' in data:
        data = ' '.join(data.split(' ')[1:])
    try:
        trns = translator.translate(data, src='fr').text
    except:
        trns = 'N/A'
    return trns

codebook = pd.io.stata.StataReader(filename).variable_labels()
codebook_df = pd.DataFrame({'key':list(codebook.keys()), 'description':list(codebook.values())}) \
    .set_index('key')


#print('\n'.join(list(codebook.values())))

#codebook_df['translation'] = codebook_df['description'].apply(translate)
#codebook_df.to_csv('codebook_trans.csv')

# Age
#print(codebook['w102']) # 102 Age

# Parity
#print(codebook['w208']) # 208 Nombre total d'enfants nés

print(data['method'].unique().tolist()) # ['No method', 'Injectables', 'Natural methods', 'Daily pill', 'Implants', 'Other traditional method', 'Male condom', 'Breastfeeding/LAM', 'iucd', 'Female sterilization', 'Female condom', 'Other modern method', 'Emergency pill']

print(data['methodtype'].unique().tolist()) # ['No method', 'Modern', 'Traditional']

print(data['age5'].unique().tolist()) # ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']
