#%%
from load_csv import df

import matplotlib.pyplot as plt
import seaborn as sns

sns.lineplot(data=df) 
plt.show()

  
#%%
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


#%%
# Split the data into X (independent variable) and y (dependent variable)
X = df['Date']
y = df['Infections']

# Add a constant term to the independent variable
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X)
results = model.fit()

# Print the summary of the regression results
print(results.summary())

#%%
# Get the predicted values
y_pred = results.predict(X)
print(y_pred)

#%%
# Plot the data points and the regression line
plt.scatter(df['Date'], y, color='blue', label='Actual')
plt.plot(df['Date'], y_pred, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Target Variable')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

#%%
# Residual Analysis
residuals = y - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

sns.histplot(residuals)
plt.show()

sns.boxplot(x=residuals)
plt.show()

#%%
# Residual Analysis
# Step 3: Calculate squared residuals
squared_residuals = residuals ** 2
# Step 4: Compute Cox distance
cox_distance = abs(residuals / (squared_residuals.mean()**0.5))
# Step 5: Plot histogram and boxplot
sns.histplot(cox_distance)
plt.show()

sns.boxplot(x=cox_distance)
plt.show()

# %%
# Perform Anderson-Darling test
from scipy.stats import anderson

result = anderson(residuals)

# Print the test result
print("Anderson-Darling Test:")
print(f"Statistic: {result.statistic}")
print(f"Critical Values: {result.critical_values}")
for i in range(len(result.significance_level)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < cv:
        print(f"The residuals likely follow a normal distribution at the {sl*100}% significance level.")
    else:
        print(f"The residuals do not follow a normal distribution at the {sl*100}% significance level.")
#%%

#%%
from sklearn.linear_model import LinearRegression

# Split the data into X (independent variable) and y (dependent variable)
X = df['Date'].values.reshape(-1, 1)
y = df['Infections'].values

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)
print(y_pred)

#%%
# Calculate the R-squared value
r_squared = model.score(X, y)
print("R-squared:", r_squared)

#%%
# Plot the data points and the regression line
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Target Variable')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

# Residual Analysis
residuals = y - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()
# %%
sns.histplot(residuals)
plt.show()

sns.boxplot(x=residuals)
plt.show()

#%%
# Residual Analysis
# Step 3: Calculate squared residuals
squared_residuals = residuals ** 2
# Step 4: Compute Cox distance
cox_distance = abs(residuals / (squared_residuals.mean()**0.5))
# Step 5: Plot histogram and boxplot
sns.histplot(cox_distance)
plt.show()

sns.boxplot(x=cox_distance)
plt.show()

# %%
# Perform Anderson-Darling test
from scipy.stats import anderson

result = anderson(residuals)

# Print the test result
print("Anderson-Darling Test:")
print(f"Statistic: {result.statistic}")
print(f"Critical Values: {result.critical_values}")
for i in range(len(result.significance_level)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < cv:
        print(f"The residuals likely follow a normal distribution at the {sl*100}% significance level.")
    else:
        print(f"The residuals do not follow a normal distribution at the {sl*100}% significance level.")
# %%
