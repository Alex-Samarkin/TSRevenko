# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file = "COVID_PSK.csv"

df = pd.read_csv(file,sep=';',encoding='utf-8',index_col='Date')

df["inf_day"] = df["Infections"].diff(periods=1)

print(df.head())

print(df.columns.tolist())

#%%
# Assuming your DataFrame is called 'df'
df['Date'] = df.index


#%%
# Convert Date column to numerical format
df['Date'] = pd.to_datetime(df['Date'],format='%d.%m.%Y',dayfirst=True)
print(df['Date'])

#%%
# Convert Date column to numerical format and convert from nanoseconds to days
df['Date'] = pd.to_numeric(df['Date']-df['Date'].min()) / (24*60*60*1000000000)
print(df['Date'])


#%%
if __name__ == '__main__':
    #plot the data using seaborn versus Date
    fig, axes = plt.subplots(2,1, figsize=(10, 8))

    sns.lineplot(data= df, ax =axes[0])
    sns.lineplot(data= df, x="Date", y="inf_day", ax =axes[1])
    plt.show()

    print("Save to csv for using next time1")    
    df.to_csv("df.csv")
