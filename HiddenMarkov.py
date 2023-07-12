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


#%%
# https://www.statsmodels.org/dev/examples/notebooks/generated/markov_regression.html
# https://hmmlearn.readthedocs.io/en/0.2.0/auto_examples/plot_hmm_stock_analysis.html
import statsmodels.api as sm
df1 = df[['Date','Infections']]

#%%

from hmmlearn import hmm
# Define the number of hidden states
n_states = 5
# Create an HMM model object
model = hmm.GaussianHMM(n_components=n_states,n_iter=1000,algorithm='viterbi')

# Fit the model
model.fit(df1)
print(f'Converged: {model.monitor_.converged}')

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(df1)
print(f"Optimal sequence of internal hidden state: {hidden_states}")

# %%
# print model summary
print(model)
print(model.transmat_)
print(model.n_iter)

#%%
states = model.predict(df1)

#%%
fig, ax = plt.subplots()
ax.plot(df1['Date'],model.transmat_[states], ".-", ms=6, mfc="orange")
plt.show()

#%%
n_states = 5
# Create an HMM model object
model = hmm.GMMHMM(n_components=n_states,n_iter=1000,algorithm='viterbi')

# Fit the model
model.fit(df1)
print(f'Converged: {model.monitor_.converged}')

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(df1)
print(f"Optimal sequence of internal hidden state: {hidden_states}")

plt.plot(df1['Date'], hidden_states)
plt.show()

# %%
# print model summary
print(model)
print(model.transmat_)
print(model.n_iter)

#%%
states = model.predict(df1)

#%%
fig, ax = plt.subplots()
ax.plot(df1['Date'],model.transmat_[states], ".-", ms=6, mfc="orange")
plt.show()