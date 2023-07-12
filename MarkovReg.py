#%%
import pandas as pd
import numpy as np
import pmdarima as pm
import matplotlib.pyplot as plt
from scipy import stats

#%%
from load_csv import df
# clear all indexes
df = df.reset_index(drop=True)
# df = df.set_index('Date')


# %%
# https://www.statsmodels.org/dev/examples/notebooks/generated/markov_regression.html
import statsmodels.api as sm
df1 = df[['Date','Infections']]
df1['Inf_day'] = df1['Infections'].diff().fillna(df1['Infections'])
k_reg = 4

# Fit the model
# (a switching mean is the default of the MarkovRegession model)
mod = sm.tsa.MarkovRegression(df1['Infections'], k_regimes=k_reg)
res = mod.fit()

mod1 = sm.tsa.MarkovRegression(df1['Inf_day'], k_regimes=k_reg)
res1 = mod1.fit()

#%%
print(res.summary())

for i in range(k_reg):
    res.smoothed_marginal_probabilities[i].plot(title=f"{i}")
    plt.show()    
    
for i in range(k_reg):
    res.smoothed_marginal_probabilities[i].plot(title=f"{k_reg}")
plt.show()

#%%
print(res1.summary())

for i in range(k_reg):
    res.smoothed_marginal_probabilities[i].plot(title=f"{i}")
    plt.show()    
    
for i in range(k_reg):
    res.smoothed_marginal_probabilities[i].plot(title=f"{k_reg}")
plt.show()


# %%
# Fit the model
# (a switching mean is the default of the MarkovRegession model)
df1['Inf_day'] = df1['Infections'].diff().fillna(df1['Infections'])

#%%
mod = sm.tsa.MarkovRegression(df1['Inf_day'], k_regimes=k_reg)
res = mod.fit()

# %%

print(res.summary())

for i in range(k_reg):
    res.smoothed_marginal_probabilities[i].plot(title=f"{i}")
    plt.show()    
    
for i in range(k_reg):
    res.smoothed_marginal_probabilities[i].plot(title=f"{k_reg}")
plt.show()

# %%
