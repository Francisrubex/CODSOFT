import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
advertising = pd.read_csv("C:\\Users\\franc\\Downloads\\archive (3)\\Advertising.csv")

# Basic inspection
print(advertising.head())
print("Shape:", advertising.shape)
print(advertising.info())
print(advertising.describe())

# Check for missing values
missing_percentage = advertising.isnull().sum() * 100 / advertising.shape[0]
print("Missing Values (%):\n", missing_percentage)

# Outlier analysis using boxplots
fig, axs = plt.subplots(3, figsize=(6, 6))
sns.boxplot(x=advertising['TV'], ax=axs[0])
axs[0].set_title('TV')
sns.boxplot(x=advertising['Newspaper'], ax=axs[1])
axs[1].set_title('Newspaper')
sns.boxplot(x=advertising['Radio'], ax=axs[2])
axs[2].set_title('Radio')
plt.tight_layout()
plt.show()

# Boxplot for Sales
sns.boxplot(x=advertising['Sales'])
plt.title('Sales')
plt.show()

# Scatter plot pair relationships
sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()

# Heatmap for correlation matrix
sns.heatmap(advertising.corr(), cmap="YlGnBu", annot=True)
plt.title("Correlation Matrix")
plt.show()

# Simple Linear Regression - TV vs Sales
X = advertising['TV']
y = advertising['Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)

# Add constant term for intercept
X_train_sm = sm.add_constant(X_train)

# Fit linear regression model
lr = sm.OLS(y_train, X_train_sm).fit()
print(lr.params)
print(lr.summary())

# Regression line on training data
intercept = lr.params['const']
slope = lr.params['TV']

plt.scatter(X_train, y_train)
plt.plot(X_train, intercept + slope * X_train, 'r')
plt.title('Fitted Line on Training Data')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()

# Residuals
y_train_pred = lr.predict(X_train_sm)
res = y_train - y_train_pred

# Residual distribution
sns.histplot(res, bins=15, kde=True)
plt.title('Error Terms Distribution')
plt.xlabel('Residuals (y_train - y_pred)')
plt.show()

# Residuals vs TV
plt.scatter(X_train, res)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs TV')
plt.xlabel('TV')
plt.ylabel('Residuals')
plt.show()

# Predict on test data
X_test_sm = sm.add_constant(X_test)
y_pred = lr.predict(X_test_sm)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")

# Regression line on test data
plt.scatter(X_test, y_test)
plt.plot(X_test, intercept + slope * X_test, 'r')
plt.title('Fitted Line on Test Data')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()
