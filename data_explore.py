import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

air_data_df = pd.read_excel('./air_quality_datasets/A_1.xlsx')
print(air_data_df)
print("column headers", air_data_df.columns)
print("Shape", air_data_df.shape)

df_sel = air_data_df.loc[200:240, ['PM2_5', 'PM10', 'SO2', 'O3', 'NOX']]
print('df_100\n', df_sel)

df_noNaN = df_sel[df_sel.notnull().sum(axis=1) == 5]

# df_100 = df_100.replace(np.nan, 0)
print('===== no nan\n', df_noNaN)
