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
train = 0.9
tess = 1.0-train
train_size = int(len(df) * train)  # 90% for training, 20% for testing
train_data, test_data = df[:train_size], df[train_size:]

#%%
model = pm.auto_arima(train_data['Infections'], seasonal=True, trace=True, suppress_warnings=True)

#%%
print(model.summary() )


#%%
predictions, conf_int = model.predict(n_periods=len(test_data), return_conf_int=True)
predicted_df = pd.DataFrame(predictions, index=test_data.index, columns=['Predicted Y'])

print(predicted_df)

#%%
from statsmodels.stats.diagnostic import acorr_ljungbox
residuals = model.resid()
print("Ljung-Box Test:")
ljung = acorr_ljungbox(residuals, lags=360)
print(ljung)
plt.plot(ljung)
plt.show()
# print(f"Ljung-Box Q-statistic={lbvalue}, p-value={pvalue}")

#%%
plt.plot(train_data.index, train_data['Infections'], label='Train')
plt.plot(test_data.index, test_data['Infections'], label='Test')
plt.plot(predicted_df.index, predicted_df['Predicted Y'], label='Predicted')
plt.fill_between(predicted_df.index, conf_int[:, 0], conf_int[:, 1], alpha=0.2, color='gray', label='Confidence Interval')
plt.legend()
plt.show()

# %%
from statsmodels.tsa.seasonal import STL
df1 = df[['Date','Infections']]
ssn=[3, 7, 15, 30, 60, 90, 120, 180, 240 , 360]
for tmp in ssn:
    res = STL(df1['Infections'],period=tmp).fit()
    fig = res.plot()
    plt.show()
