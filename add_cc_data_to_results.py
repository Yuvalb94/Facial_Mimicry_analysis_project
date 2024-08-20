import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

origin = r'C:\Users\yuval\OneDrive\Desktop\לימודים\שנה ג\סדנת מחקר\final task'
os.chdir(origin)

cc_res_filename = r'CC_results_new_zscore_cosineSim_thrd_0_05_400ppw.csv'
cc_results = pd.read_csv(cc_res_filename)
test_res_filename = r'female_particiapnts_analyzed_withChoice_16_4_30First.csv'
test_results = pd.read_csv(test_res_filename)
saving_filename = 'results_combined_new_Zscore_CosineSim_thrd_0_05_400_ppw.csv'
print(cc_results)
print(test_results)
#
merged = test_results.merge(cc_results)
print(merged)
merged.to_csv(saving_filename, index=False)

