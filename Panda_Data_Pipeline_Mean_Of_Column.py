# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 23:19:08 2019

@author: Admin
"""

import pandas as pd

# create empty dataframe
df = pd.DataFrame()

# Create columns

df['name'] = ['Damian', 'Sarah', 'Ray']

df['employed'] = ['Yes', 'Yes', 'No'] 

df['age'] = [32, 29, 19]

# view dataframe
df

# create a function to calculate the mean of each column
def mean_age_by_group(dataframe, col):
    # groups the data by a column and return the mean age per group
    return dataframe.groupby(col).mean()

# create a function to change column headers to UPPERCASE
def uppercase_column_name(dataframe):
    # change all column headers to UPPERCASE
    dataframe.columns = dataframe.columns.str.upper()

    # return dataframe
    return dataframe


# create a pipeline that applies both functions above
(df.pipe(mean_age_by_group, col = 'employed')
    # then apply the uppercase function
    .pipe(uppercase_column_name)    
)

