#%%
import pandas as pd 
df = pd.read_csv("AirPassengers.csv")

print(df.head())

#%%
print(df.tail())

#%%
print(df.describe())

#%%
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
print(df.head())

#%%
df.set_index('Month', inplace=True)
print(df.head())

#%%
import matplotlib.pyplot as plt
import seaborn as sns 

sns.lineplot(data=df)
plt.show()
# %%
rolling_mean = df.rolling(7).mean()
rolling_std = df.rolling(7).std()
# %%
plt.plot(df, color="blue",label="Original Passenger Data")
plt.plot(rolling_mean, color="red", label="Rolling Mean Passenger Number")
plt.plot(rolling_std, color="black", label = "Rolling Standard Deviation in Passenger Number")

plt.legend(loc="best")
# %%
from statsmodels.tsa.stattools import adfuller
# %%
adft = adfuller(df,autolag="AIC")
# %%
output_df = pd.DataFrame({"Values":[adft[0],adft[1],adft[2],adft[3], adft[4]['1%'], adft[4]['5%'], adft[4]['10%']]  , 
                          "Metric":["Test Statistics","p-value","No. of lags used","Number of observations used", "critical value (1%)", "critical value (5%)", "critical value (10%)"]})
print(output_df)
# %%
autocorrelation_lag1 = df['#Passengers'].autocorr(lag=1)
print("One Month Lag: ", autocorrelation_lag1)
# %%
autocorrelation_lag3 = df['#Passengers'].autocorr(lag=3)
print("Three Month Lag: ", autocorrelation_lag3)

autocorrelation_lag6 = df['#Passengers'].autocorr(lag=6)
print("Six Month Lag: ", autocorrelation_lag6)

autocorrelation_lag9 = df['#Passengers'].autocorr(lag=9)
print("Nine Month Lag: ", autocorrelation_lag9)
# %%
from statsmodels.tsa.seasonal import seasonal_decompose
# %%
decompose = seasonal_decompose(df['#Passengers'],model='additive', period=7)
decompose.plot()
plt.show()
# %%
df['Date'] = df.index
train = df[df['Date'] < pd.to_datetime("1960-08", format='%Y-%m')]
train['train'] = train['#Passengers']
del train['Date']
del train['#Passengers']
test = df[df['Date'] >= pd.to_datetime("1960-08", format='%Y-%m')]
del test['Date']
test['test'] = test['#Passengers']
del test['#Passengers']
plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.title("Train/Test split for Passenger Data")
plt.ylabel("Passenger Number")
plt.xlabel('Year-Month')
sns.set()
plt.show()
# %%
######
# conda install -c conda-forge pmdarima
#
from pmdarima.arima import auto_arima
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)
forecast = model.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])
# %%
plt.plot(forecast, color = "black")
# %%
