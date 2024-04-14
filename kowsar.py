'''
This program is intended to clean the data but also answer the following questions:
Q1: Explore what is measured under smartphone ratings. Try to find out if there is a correlation between price, tech specifications, and user ratings. We can use correlation analysis or linear regression for this.
Q2: Does 5G affect smartphone prices? Analyze the adoption rate of 5G technology in smartphones over price segments. 
Q3: Analyze the smartphone market distribution by brand, price segments, and key features. We can compare distributions, make distribution plots and pie charts etc. Which brand offers the most 5G-enabled devices? Which brand is produces the most expensive smartphones on average? What is the average rating per brand?
Q4: Develop a model that predicts the price of a smartphone based on its specifications.
Q5: What is the average lifespan of smartphones based on their specifications?
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import seaborn as sns
from scipy.stats import ttest_ind
import statsmodels.api as sm

'''
Q2) Analyze the adoption rate of 5G technology in smartphones over price segments.
we have used different analysis method for this question and they are indicated below
'''
# statistic description
print(df.groupby('has_5g')['price'].describe())

# proportion of 5G in each price segment
df['price_segment'] = pd.cut(df['price'], bins=[0, 5000, 10000, 15000, 20000, 30000, np.inf], labels=['<5000', '5000-10000', '10000-15000', '15000-20000', '20000-30000', '>30000'])
grouped_data = df.groupby(['price_segment', 'has_5g'], observed=True).size().unstack().fillna(0)
print(grouped_data)

# visualization
sns.boxplot(x='has_5g', y='price', data=df)
plt.title('5G and prices')
plt.xlabel('has 5G')
plt.ylabel('Prices')
plt.savefig('5G price distribution')

# correlation
print(np.corrcoef(df['price'], df['has_5g'])[0, 1])

# regression analysis

# converting boolean to 0/1
df['has_5g'] = df['has_5g'].astype(int)
X = sm.add_constant(df['has_5g'])  # adding a constant
model = sm.OLS(df['price'], X).fit()
print(model.summary())

# T-Test
group_5g = df[df['has_5g'] == 1]['price']
group_non_5g = df[df['has_5g'] == 0]['price']
t_stat, p_val = ttest_ind(group_5g, group_non_5g, equal_var=False)
print(f"T-statistic: {t_stat}, P-value: {p_val}")


'''
Q5: What is the average lifespan of smartphones based on their specifications?
'''
# Selecting these features as the specification where they can affect the lifespan
features_for_lifespan = df[['battery_capacity', 'processor_speed', 'ram_capacity', 'screen_size']]
target_for_lifespan = df['rating']

# Drop any rows with missing values
features_for_lifespan = features_for_lifespan.dropna()
target_for_lifespan = target_for_lifespan.loc[features_for_lifespan.index]

# Splitting the data into training and testing sets
X_train_lifespan, X_test_lifespan, y_train_lifespan, y_test_lifespan = train_test_split(
    features_for_lifespan, target_for_lifespan, test_size=0.2, random_state=42)

# Initializing and training a linear regression model
model_lifespan = LinearRegression()
model_lifespan.fit(X_train_lifespan, y_train_lifespan)

# Predicting and evaluating the model
y_pred_lifespan = model_lifespan.predict(X_test_lifespan)
mse_lifespan = mean_squared_error(y_test_lifespan, y_pred_lifespan)
r2_lifespan = r2_score(y_test_lifespan, y_pred_lifespan)

# Output the results
print("Coefficients for Lifespan Estimation Model:", model_lifespan.coef_)
print("Mean Squared Error for Lifespan Model:", mse_lifespan)
print("R2 Score for Lifespan Model:", r2_lifespan)