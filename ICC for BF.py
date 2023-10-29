#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 13:59:10 2023

@author: mr.basix
"""
import pandas as pd
import numpy as np
import pingouin as pg
import statsmodels as sms
import matplotlib.pyplot as plt
import seaborn as sns


#Subject 1 data
#caliper_sum_subject1=(176,184,178,162,173,166,166, None, None)
#Ultrasound_sum_subject1=(76.3,73.9,74,72.6,70,70.8,72.5,71.6,None,None)
#bodyfat_percent__Caliper_subject1=(26.5,27.3,26.1,24.9,26.1,25.3,25.3,26.0,None,None)
#bodyfat_percent__Ultrasound_subject1=(23.5,23,23.1,22.7,22,22.2,22.5,22.3, None, None)

#Subject 2 data
#caliper_sum_subject2=(74,73,82,82,76,72,83,82,74,73)
#Ultrasound_sum_subject2=(41.3,43.5,48.5,49.8,47,43.2,49.9,51,48.5,49.8)
#bodyfat_percent__Caliper_subject2=(15.8,15.7,17.2,17.2,16.2,15.5,17.3,17.2,15.8,15.7)
#bodyfat_percent__Ultrasound_subject2=(17.3,18,19.6,20.1,19.1,17.9,20.1,20.4,19.6,20.1)

#Subject 3 data
#caliper_sum_subject3=(127,124,114,113,117,118,127,129,118,112)
#Ultrasound_sum_subject3=(67.6,65.2,61.5,59.8,67,62.6,64.3,69,61.6,61.2)
#bodyfat_percent__Caliper_subject3=(17.3,16.9,15.6,15.4,16.0,16.1,17.3,17.6,16.1,15.3)
#bodyfat_percent__Ultrasound_subject3=(17.9,17.3,16.8,16.3,17.9,17.1,17.7,18.4,16.8,16.7)

#Subject 4 data
#caliper_sum_subject4=(116,110,105,105,121,126,122,126,119,118)
#Ultrasound_sum_subject4=(61.9,64.4,66.9,63.3,56.7,56.3,50.2,59.6,65.9,67.1)
#bodyfat_percent__Caliper_subject4=(15.7,14.9,14.2,14.2,16.4,17.0,16.5,17.0,16.1,16.0)
#bodyfat_percent__Ultrasound_subject4=(16.8,17.4,18,17.1,15.4,15.3,16,16.1,17.3,17.5)

#Subject 5 data
#caliper_sum_subject5=(135,141,136,139,142,144,154,154,148,152)
#Ultrasound_sum_subject5=(67.4,67.5,76.2,75.1,73.6,75.3,68.3,72.9,74.2,71.9)
#bodyfat_percent__Caliper_subject5=(25.7,26.6,25.8,26.3,26.7,27.0,28.4,28.4,27.5,28.1)
#bodyfat_percent__Ultrasound_subject5=(25.7,25.7,28,27.7,27.3,27.5,25.8,27.1,27.5,27.0)


###


# Initialize empty list to collect DataFrames
df_list = []


# Define measurements
measures = ['caliper_sum', 'Ultrasound_sum', 'bodyfat_percent__Caliper', 'bodyfat_percent__Ultrasound']

# Data for each subject and measurement type
data = {
    'caliper_sum': [(176, 184, 178, 162, 173, 166, 166, None, None),
                    (74, 73, 82, 82, 76, 72, 83, 82, 74, 73),
                    (127, 124, 114, 113, 117, 118, 127, 129, 118, 112),
                    (116, 110, 105, 105, 121, 126, 122, 126, 119, 118),
                    (135, 141, 136, 139, 142, 144, 154, 154, 148, 152)
    ],
    'Ultrasound_sum': [
        (76.3, 73.9, 74, 72.6, 70, 70.8, 72.5, 71.6, None, None),
        (41.3, 43.5, 48.5, 49.8, 47, 43.2, 49.9, 51, 48.5, 49.8),
        (67.6, 65.2, 61.5, 59.8, 67, 62.6, 64.3, 69, 61.6, 61.2),
        (61.9, 64.4, 66.9, 63.3, 56.7, 56.3, 50.2, 59.6, 65.9, 67.1),
        (67.4, 67.5, 76.2, 75.1, 73.6, 75.3, 68.3, 72.9, 74.2, 71.9)
    ],
    'bodyfat_percent__Caliper': [
        (26.5, 27.3, 26.1, 24.9, 26.1, 25.3, 25.3, 26.0, None, None),
        (15.8, 15.7, 17.2, 17.2, 16.2, 15.5, 17.3, 17.2, 15.8, 15.7),
        (17.3, 16.9, 15.6, 15.4, 16.0, 16.1, 17.3, 17.6, 16.1, 15.3),
        (15.7, 14.9, 14.2, 14.2, 16.4, 17.0, 16.5, 17.0, 16.1, 16.0),
        (25.7, 26.6, 25.8, 26.3, 26.7, 27.0, 28.4, 28.4, 27.5, 28.1)
    ],
    'bodyfat_percent__Ultrasound': [
        (23.5, 23, 23.1, 22.7, 22, 22.2, 22.5, 22.3, None, None),
        (17.3, 18, 19.6, 20.1, 19.1, 17.9, 20.1, 20.4, 19.6, 20.1),
        (17.9, 17.3, 16.8, 16.3, 17.9, 17.1, 17.7, 18.4, 16.8, 16.7),
        (16.8, 17.4, 18, 17.1, 15.4, 15.3, 16, 16.1, 17.3, 17.5),
        (25.7, 25.7, 28, 27.7, 27.3, 27.5, 25.8, 27.1, 27.5, 27.0)
    ]
}

# Loop over each measure type
for measure in measures:
    # Loop over each subject
    for subject_idx, subject_data in enumerate(data[measure], 1):
        temp_df = pd.DataFrame({
            'Subject': [f'Subject_{subject_idx}'] * len(subject_data),
            'Session': list(range(1, len(subject_data) + 1)),
            'MeasurementType': [measure] * len(subject_data),
            'Value': subject_data
        })
        df_list.append(temp_df)

# Concatenate all the temporary DataFrames
df = pd.concat(df_list, ignore_index=True)

# Drop rows with missing data
df.dropna(inplace=True)

# Initialize an empty DataFrame to store ICC and MDD results
results_df = pd.DataFrame(columns=['Measure', 'ICC1k', 'MDD'])

# Loop over each measure type and calculate ICC
for measure in measures:
    sub_df = df[df['MeasurementType'] == measure]
    try:
        icc_result = pg.intraclass_corr(data=sub_df, targets='Subject', raters='Session', ratings='Value', nan_policy='omit').round(3)
        print(f"ICC for {measure}:\n{icc_result}\n")
        
        # Formatting p-value
        icc_result['pval'] = icc_result['pval'].apply(lambda x: "{:.4f}".format(x))
        print(f"Formatted ICC for {measure}:\n{icc_result}\n")
        sd_value = np.std(sub_df['Value'])
     
        
        # Calculate MDD for ICC1,k
        icc_value_1k = icc_result.loc[icc_result['Type'] == 'ICC1k', 'ICC'].values[0]
        sem_value_1k = sd_value * np.sqrt(1 - icc_value_1k)
        mdd_value_1k = sem_value_1k * np.sqrt(2) * 1.96
        print(f"MDD for {measure} based on ICC1,k: {mdd_value_1k}\n")
        
    except ValueError as e:
        print(f"Could not calculate ICC for {measure} due to the following error:\n{e}\n")




    
# Plot the table




