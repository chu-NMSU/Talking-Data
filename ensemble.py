import pandas as pd
import numpy as np

df1 = pd.read_csv('output/submit_v0_5_1.csv', dtype={'device_id':str})
df2 = pd.read_csv('output/kaggle_xgb_text_nofill.csv', dtype={'device_id':str})
df_mean = df1
group_list = ['F24-26','F27-28','M29-31','M22-','F33-42','M32-38','M39+','F43+',\
        'F23-','M27-28','M23-26','F29-32']
df_mean[group_list] = (df1[group_list] + df2[group_list])/2
df_mean.to_csv('output/ensemble_avg.csv', index=False)
