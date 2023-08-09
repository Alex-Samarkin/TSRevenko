# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file = "COVID_PSK.csv"

#df = pd.read_csv(file,sep=';',encoding='utf-8',index_col='Date')

#df["inf_day"] = df["Infections"].diff(periods=1)

#print(df.head())

#print(df.columns.tolist())

#%%
# Assuming your DataFrame is called 'df'
#df['Date'] = df.index


#%%
# Convert Date column to numerical format
#df['Date'] = pd.to_datetime(df['Date'],format='%d.%m.%Y',dayfirst=True)

#%%
# Convert Date column to numerical format and convert from nanoseconds to days
#df['Date'] = pd.to_numeric(df['Date']-df['Date'].min()) / (24*60*60*1000000000)

def QuickLoad(file = "COVID_PSK.csv"):
    df = pd.read_csv(file,sep=';',encoding='utf-8',index_col='Date')
    df["inf_day"] = df["Infections"].diff(periods=1)
    df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date'],format='%d.%m.%Y',dayfirst=True)
    df = df.reset_index(drop=True)
    df1 = df[['Date','Infections']]
    df1.reset_index(drop=True)
    df1['Inf_day'] = df1['Infections'].diff().fillna(df1['Infections'])
    l = df1['Date']
    ts = df1['Inf_day']
    data = ts.values
    X = data.reshape(data.shape[0])
    return X,l,ts,df1,df


#%%
if __name__ == '__main__':
    QuickLoad(file = "COVID_PSK.csv")
    #plot the data using seaborn versus Date
    fig, axes = plt.subplots(2,1, figsize=(10, 8))

    sns.lineplot(data= df, ax =axes[0])
    axes[0].tick_params(bottom=False)
    axes[0].set(xticklabels=[])
    axes[0].set(xlabel=None)
    sns.lineplot(data= df, x="Date", y="inf_day", ax =axes[1])
    plt.show()

    print("Save to csv for using next time1")    
    df.to_csv("df.csv")
