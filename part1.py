#
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

'''
Q1a) Cleaning the data
'''

print("Data Analysis to answer question 1: ")

df = pd.read_csv('smartphone_cleaned_v5.csv')

# Calculate percentage of missing values for each column
missing_percentages = df.isnull().sum() / len(df) * 100

# Set a threshold for dropping columns so that columns with more than 20% missing values will be dropped
threshold = 20  
columns_to_drop = missing_percentages[missing_percentages > threshold].index

# Drop columns (2 columns in this case)
df = df.drop(columns_to_drop, axis=1)

'''
Q1b) Performing Correlation Analysis
'''

# Calculate correlation matrix only for numeric columns
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()

# Plotting the heatmap
plt.figure(figsize=(20, 15))  # You can adjust these numbers as needed
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.subplots_adjust(left=0.2, bottom=0.4)
plt.show()




# #loading the dataset and examining the structure
# df = pd.read_csv('smartphone_cleaned_v5.csv')
# #print (df.head())
# #df.info()
# # couting total number of null and unknwon values for the report
# null_counts = df.isnull().sum()
# unknown_counts = (df == 'Unknown').sum()
# combined_counts = pd.DataFrame({'Null Values': null_counts, 'Unknown Values': unknown_counts})
# #print(combined_counts)

# #check and handle for missing values and 
# # Fill missing numerical values with the median
# df['rating'] = df['rating'].fillna(df['rating'].median())
# df['processor_speed'] = df['processor_speed'].fillna(df['processor_speed'].median())

# # Fill missing categorical values with 'Unknown' or the most common value
# df['processor_brand'] = df['processor_brand'].fillna('Unknown')
# df['os'] = df['os'].fillna(df['os'].mode()[0])
# df['extended_upto'] = df['extended_upto'].fillna(0)

# # Impute missing numerical values with the median
# df['num_cores'] = df['num_cores'].fillna(df['num_cores'].median())
# df['battery_capacity'] = df['battery_capacity'].fillna(df['battery_capacity'].median())
# df['num_front_cameras'] = df['num_front_cameras'].fillna(df['num_front_cameras'].median())
# df['primary_camera_front'] = df['primary_camera_front'].fillna(df['primary_camera_front'].median())

# # For 'fast_charging', assuming it's categorical
# df['fast_charging'] = df['fast_charging'].fillna('Unknown') 

# # For 'extended_upto', consider logic based on 'extended_memory_available'
# df.loc[df['extended_memory_available'] == 0, 'extended_upto'] = df.loc[df['extended_memory_available'] == 0, 'extended_upto'].fillna(0)

# # Check the updated summary to confirm changes
# #print(df.isnull().sum())

'''
his part of code for investigating more into what is measured under smartphone ratings. 
Try to find out if there is a correlation between price, tech specifications, and user ratings.
We can use correlation analysis and linear regression for this.
Distribution of user rating
'''
df['rating'].hist()
plt.title('Distribution of User Ratings')
plt.xlabel('User Ratings')
plt.ylabel('Frequency')
plt.savefig('distribution_of_user_rating.png')
numeric_df =  df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
#print(df.dtypes)
#print(corr_matrix['rating'])
'''
1 step further using linear regression to not only identify the relationships but also
quantify the impact of one or more prediction on an ourcome variable
'''
predictors = ['processor_speed', 'refresh_rate', 'primary_camera_front', 'price']
target = 'rating'
X = df[predictors]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print(f'Mean Squared Error is : {mse:.2f}')
# print(f'R^2 Score is : {r2:.2f}')


# for this part we want to find some answers about "Does 5G affect smartphone prices? Analyze the adoption rate of 5G technology in smartphones over price segments."

# statistic description
print(df.groupby('has_5g')['price'].describe())


#proportion of 5G in each price segment
df['price_segment'] = pd.cut(df['price'], bins=[0, 5000, 10000, 15000, 20000, 30000, np.inf], labels=['<5000', '5000-10000', '10000-15000', '15000-20000', '20000-30000', '>30000'])
grouped_data = df.groupby(['price_segment', 'has_5g'], observed=True).size().unstack().fillna(0)
print(grouped_data)

#visualization
sns.boxplot(x='has_5g', y='price', data=df)
plt.title('5G and prices')
plt.xlabel('has 5G')
plt.ylabel('Prices')
plt.savefig('5G price distribution')

#correlation
print(np.corrcoef(df['price'], df['has_5g'])[0, 1])

#regression analysis
import statsmodels.api as sm
# converting boolean to 0/1
df['has_5g'] = df['has_5g'].astype(int)
X = sm.add_constant(df['has_5g'])  # adding a constant
model = sm.OLS(df['price'], X).fit()
print(model.summary())

#T-Test
group_5g = df[df['has_5g'] == 1]['price']
group_non_5g = df[df['has_5g'] == 0]['price']
t_stat, p_val = ttest_ind(group_5g, group_non_5g, equal_var=False)
print(f"T-statistic: {t_stat}, P-value: {p_val}")

'''
Q4a) Splitting data into training and testing sets
'''

X = df.drop('price', axis=1)  # Features (all columns except 'price')
y = df['price']  # Target variable

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Identify categorical variables
categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
print("Categorical columns: ", categorical_cols)

# Apply one-hot encoding to the training and testing data
X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Ensure both training and testing data have the same dummy variables
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Check the data types and ensure that all are numeric
print(X_train.info())
print(X_test.info())

# Create an imputer object with a strategy of filling with the median
imputer = SimpleImputer(strategy='median')

# Fit the imputer on your training data and transform both training and test data
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Convert the imputed arrays back to DataFrames
X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns, index=X_test.index)

'''
Q4b) Linear Regression Model
'''

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Plotting actual vs. predicted values for price
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # A diagonal line where predicted = actual
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted Smartphone Prices')
plt.show()
