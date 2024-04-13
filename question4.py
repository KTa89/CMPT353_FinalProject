import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

df = pd.read_csv('smartphone_cleaned_v5.csv')

# print("in the beginning", (df.head()))

# Calculate percentage of missing values for each column
missing_percentages = df.isnull().sum() / len(df) * 100
# print(missing_percentages)

# Set a threshold for dropping columns
threshold = 20
columns_to_drop = missing_percentages[missing_percentages > threshold].index

# Drop columns
df = df.drop(columns_to_drop, axis=1)

# Calculate correlation matrix only for numeric columns
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()

# Plotting the heatmap
plt.figure(figsize=(20, 15))  # You can adjust these numbers as needed
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.subplots_adjust(left=0.2, bottom=0.4)
plt.show()

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

# Optional: Convert the imputed arrays back to DataFrames
X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns, index=X_test.index)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Plotting actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # A diagonal line where predicted = actual
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted Smartphone Prices')
plt.show()





