import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats

pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None)

df = pd.read_csv('ShenyangPM20100101_20151231.csv')

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.info())
print(df.isnull().sum() / df.shape[0] * 100)

df.drop(['PM_Taiyuanjie'], axis=1, inplace=True)
df.drop(['PM_Xiaoheyan'], axis=1, inplace=True)
print(df.isnull().sum() / df.shape[0] * 100)

print(df.describe())
print(df['DEWP'].unique())
df['DEWP'] = df['DEWP'].replace(-97, np.nan)
print(df['DEWP'].unique())

print(df.describe())
print(df['precipitation'].unique())

df['DEWP'].fillna(method='bfill', inplace=True)
df['HUMI'].fillna(method='bfill', inplace=True)
df['PRES'].fillna(np.nanmean(df['PRES']), inplace=True)
df['TEMP'].fillna(method='bfill', inplace=True)
df['cbwd'].fillna(method='bfill', inplace=True)
df['Iws'].fillna(np.nanmean(df['Iws']), inplace=True)
print(df.isnull().sum() / df.shape[0] * 100)
print(df.describe())

print(df['PM_US Post'].describe())

corr_mat = df.corr()
sb.heatmap(corr_mat, annot=True)
plt.show()

print(df)

df.dropna(axis=0, inplace=True)
print(df)
print(df.isnull().sum() / df.shape[0] * 100)
print(df.describe())

df['season'] = df['season'].replace(1,'prolece')
df['season'] = df['season'].replace(2,'leto')
df['season'] = df['season'].replace(3,'jesen')
df['season'] = df['season'].replace(4,'zima')
print(df)

df_season = df.set_index('season')
print(df_season.head())
plt.figure(1)
plt.hist(df_season.loc['prolece', 'PM_US Post'], density=True, alpha=0.5, bins=50, label="prolece")
plt.hist(df_season.loc['leto', 'PM_US Post'], density=True, alpha=0.5, bins=50, label="leto")
plt.hist(df_season.loc['jesen', 'PM_US Post'], density=True, alpha=0.5, bins=50, label="jesen")
plt.hist(df_season.loc['zima', 'PM_US Post'], density=True, alpha=0.5, bins=50, label="zima")
plt.legend()

plt.figure(2)
gb = df.groupby(by=['month']).mean()
#plt.plot(np.arange(1, 13, 1), gb['PM_US Post'], 'b', label='Evropa')
plt.hist(gb.loc[1, 'PM_US Post'],  alpha=1, bins=10)
plt.hist(gb.loc[2, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[3, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[4, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[5, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[6, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[7, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[8, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[9, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[10, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[11, 'PM_US Post'],  alpha=0.5, bins=10)
plt.hist(gb.loc[12, 'PM_US Post'],  alpha=0.5, bins=10)

df_month = df.set_index('month')
nesto = gb.loc[1, 'PM_US Post']

plt.figure(3)
plt.boxplot([df_month.loc[1,'PM_US Post'], df_month.loc[2,'PM_US Post'], df_month.loc[3,'PM_US Post'], df_month.loc[4,'PM_US Post'], df_month.loc[5,'PM_US Post'], df_month.loc[6,'PM_US Post'], df_month.loc[7,'PM_US Post'], df_month.loc[8,'PM_US Post'], df_month.loc[9,'PM_US Post'], df_month.loc[10,'PM_US Post'], df_month.loc[11,'PM_US Post'], df_month.loc[12,'PM_US Post']]) 
plt.ylabel('PM')
plt.xlabel('Mesec')
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12], ["Jan", "Feb", "Mart", "Apr", "Maj", "Jun", "Jul", "Avg", "Sep", "Okt", "Nov", "Dec"])
plt.grid()
