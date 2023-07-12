
#%%
from load_csv import df

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns


# Define the logistic function
def logistic_function(x, A, k, x0):
    return A / (1 + np.exp(-k*(x-x0)))

# Split the data into X (independent variable) and y (dependent variable)
x_data = df['Date']
y_data = df['Infections']

# Generate sample data
#x_data = np.linspace(-10, 10, 100)
#y_data = logistic_function(x_data, 1, 1, 0) + np.random.normal(0, 0.2, len(x_data))  # Adding some noise

# Perform curve fitting
initial_guess = [1, 1, 0]  # Initial parameter values
params, params_covariance = curve_fit(logistic_function, x_data, y_data, p0=initial_guess)

# Extract the fitted parameters
A_fit, k_fit, x0_fit = params

print(
"""
A: The curve's maximum value (the upper asymptote)
k: The growth rate parameter
x0: The x-value of the curve's midpoint
""")

print(f"A = {A_fit}, k = {k_fit}, x0 = {x0_fit}")


# Generate the predicted y-values using the fitted curve
y_fit = logistic_function(x_data, A_fit, k_fit, x0_fit)

# Plot the original data and the fitted curve
plt.scatter(x_data, y_data, label='Original Data')
plt.plot(x_data, y_fit, 'r-', label='Fitted Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#%%
# Residual Analysis
residuals = y_data - y_fit
plt.scatter(y_fit, residuals)
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