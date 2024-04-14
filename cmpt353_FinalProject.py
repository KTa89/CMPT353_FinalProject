#
'''
This program is intended to clean the data but also answer the following questions:
Q1: Explore what is measured under smartphone ratings. Try to find out if there is a correlation between price, tech specifications, and user ratings. We can use correlation analysis or linear regression for this.
Q2: Does 5G affect smartphone prices? Analyze the adoption rate of 5G technology in smartphones over price segments. 
Q3: Analyze the smartphone market distribution by brand, price segments, and key features. Which brand produces the most expensive smartphones on average? What is the average rating per brand?
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
plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

'''
Q2: Does 5G affect smartphone prices? Analyze the adoption rate of 5G technology in smartphones over price segments. 
'''
def analyze_5g_adoption(df):
    # Create a local copy to work with
    local_df = df.copy()

    # Convert 'has_5g' to integer locally
    local_df['has_5g'] = local_df['has_5g'].astype(int)

    # Adding price segments for local analysis
    local_df['price_segment'] = pd.cut(local_df['price'], bins=[0, 5000, 10000, 15000, 20000, 30000, np.inf], labels=['<5000', '5000-10000', '10000-15000', '15000-20000', '20000-30000', '>30000'])
    print(local_df.groupby(['price_segment', 'has_5g']).size().unstack(fill_value=0))

    # Visualize price distribution by 5G capability
    sns.boxplot(x='has_5g', y='price', data=local_df)
    plt.title('5G and Prices')
    plt.xlabel('Has 5G')
    plt.ylabel('Prices')
    plt.savefig('5GPrice.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Simple correlation and regression analysis
    print(np.corrcoef(local_df['price'], local_df['has_5g'])[0, 1])
    X = sm.add_constant(local_df['has_5g'])
    model = sm.OLS(local_df['price'], X).fit()
    print(model.summary())

    # T-test between groups
    group_5g = local_df[local_df['has_5g'] == 1]['price']
    group_non_5g = local_df[local_df['has_5g'] == 0]['price']
    t_stat, p_val = ttest_ind(group_5g, group_non_5g, equal_var=False)
    print(f"T-statistic: {t_stat}, P-value: {p_val}")

analyze_5g_adoption(df)
'''
Q3a) Splitting data into training and testing sets. Took inspiration from https://stackabuse.com/one-hot-encoding-in-python-with-pandas-and-scikit-learn/ for this part of the question.
'''
print("Data Analysis to answer question 3: ")


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


'''
# Q3b) Linear Regression Model
# '''

model = LinearRegression()
# Create an imputer object with a mean filling strategy
imputer = SimpleImputer(strategy='mean')

# Apply the imputer to your X_train dataset
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
model.fit(X_train, y_train)

# Apply the imputer to your X_test dataset
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
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
plt.savefig('linear_regression.png', dpi=300, bbox_inches='tight')
plt.close()

'''
Q4a) Price Analysis by Brand
'''
print("Data Analysis to answer question 4: ")

# Define the exchange rate from INR to CAD, to convert prices
exchange_rate = 0.017
df['price_cad'] = df['price'] * exchange_rate

price_threshold = 10000  # removing unwanted outliers
df_filtered = df[df['price_cad'] <= price_threshold]

# Count the number of models per brand
model_counts = df['brand_name'].value_counts()
valid_brands = model_counts[model_counts >= 10].index # Brands with at least 10 models
df_filtered = df[df['brand_name'].isin(valid_brands)] 

# Calculate the average price per brand
average_price_per_brand = df_filtered.groupby('brand_name')['price_cad'].mean().sort_values(ascending=False)
print(average_price_per_brand)

# Plotting using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x=average_price_per_brand.index, y=average_price_per_brand.values, palette='coolwarm')
plt.title('Average Price by Brand')
plt.xlabel('Brand')
plt.ylabel('Average Price (CAD)')
plt.xticks(rotation=45)  # Rotates the brand names for better readability
plt.savefig('price_comparison.png', dpi=300, bbox_inches='tight')
plt.close()


'''
Q4b) Rating Analysis by Brand
'''

# Calculate the average rating per brand
average_rating_per_brand = df_filtered.groupby('brand_name')['rating'].mean().sort_values(ascending=False)
print(average_rating_per_brand)

# Plotting using seaborn
plt.figure(figsize=(12, 8))
sns.barplot(x=average_rating_per_brand.index, y=average_rating_per_brand.values, palette='viridis')
plt.title('Average Rating by Brand')
plt.xlabel('Brand')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)  # Rotates the brand names for better readability
plt.savefig('rating_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

'''
Q5: What is the average lifespan of smartphones based on their specifications?
'''
print("Data Analysis to answer question 5: ")
 
# selecting these features as the specification where they can affect the lifespan
features_for_lifespan = df[['battery_capacity', 'processor_speed', 'ram_capacity', 'screen_size']]
target_for_lifespan = df['rating']
 
# check and drop any rows with missing values in either features or target
combined_df = features_for_lifespan.join(target_for_lifespan, how='inner')
cleaned_combined_df = combined_df.dropna()
 
# separate features and target after cleaning
features_for_lifespan_cleaned = cleaned_combined_df[['battery_capacity', 'processor_speed', 'ram_capacity', 'screen_size']]
target_for_lifespan_cleaned = cleaned_combined_df['rating']
 
# splitting the data into training and testing sets
X_train_lifespan, X_test_lifespan, y_train_lifespan, y_test_lifespan = train_test_split(
    features_for_lifespan_cleaned, target_for_lifespan_cleaned, test_size=0.2, random_state=42)
 
# initializing and training a linear regression model
model_lifespan = LinearRegression()
model_lifespan.fit(X_train_lifespan, y_train_lifespan)
 
# predicting and evaluating the model
y_pred_lifespan = model_lifespan.predict(X_test_lifespan)
mse_lifespan = mean_squared_error(y_test_lifespan, y_pred_lifespan)
r2_lifespan = r2_score(y_test_lifespan, y_pred_lifespan)
 
# output result
print("Coefficients for Lifespan Estimation Model:", model_lifespan.coef_)
print("Mean Squared Error for Lifespan Model:", mse_lifespan)
print("R2 Score for Lifespan Model:", r2_lifespan)
 
