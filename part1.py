#This program is intended to clean the data but also answer the following questions
#Explore what is measured under smartphone ratings. Try to find out if there is a correlation between price, tech specifications, and user ratings. We can use correlation analysis or linear regression for this.
# Does 5G affect smartphone prices? Analyze the adoption rate of 5G technology in smartphones over price segments. 
# Analyze the smartphone market distribution by brand, price segments, and key features. We can compare distributions, make distribution plots and pie charts etc.
# Develop a model that predicts the price of a smartphone based on its specifications. Need to do some research for this
# What is the average lifespan of smartphones based on their specifications?

import pandas as pd

#loading the dataset and examining the structure
df = pd.read_csv('smartphone_cleaned_v5.csv')
#print (df.head())
#df.info()
# couting total number of null and unknwon values for the report
null_counts = df.isnull().sum()
unknown_counts = (df == 'Unknown').sum()
combined_counts = pd.DataFrame({'Null Values': null_counts, 'Unknown Values': unknown_counts})
#print(combined_counts)

#check and handle for missing values and 
# Fill missing numerical values with the median
df['rating'] = df['rating'].fillna(df['rating'].median())
df['processor_speed'] = df['processor_speed'].fillna(df['processor_speed'].median())

# Fill missing categorical values with 'Unknown' or the most common value
df['processor_brand'] = df['processor_brand'].fillna('Unknown')
df['os'] = df['os'].fillna(df['os'].mode()[0])
df['extended_upto'] = df['extended_upto'].fillna(0)

# Impute missing numerical values with the median
df['num_cores'] = df['num_cores'].fillna(df['num_cores'].median())
df['battery_capacity'] = df['battery_capacity'].fillna(df['battery_capacity'].median())
df['num_front_cameras'] = df['num_front_cameras'].fillna(df['num_front_cameras'].median())
df['primary_camera_front'] = df['primary_camera_front'].fillna(df['primary_camera_front'].median())

# For 'fast_charging', assuming it's categorical
df['fast_charging'] = df['fast_charging'].fillna('Unknown') 

# For 'extended_upto', consider logic based on 'extended_memory_available'
df.loc[df['extended_memory_available'] == 0, 'extended_upto'] = df.loc[df['extended_memory_available'] == 0, 'extended_upto'].fillna(0)

# Check the updated summary to confirm changes
print(df.isnull().sum())


