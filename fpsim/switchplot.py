import pandas as pd

#------------------------------ the path to the results----------------------------------------------------#

path = "D:/APHRC/ABM/fpsim/results.csv"

results_data = pd.read_csv(path)

print(results_data.head())

#---------------------------------creating the age grpups and contraceptive use----------------------------#