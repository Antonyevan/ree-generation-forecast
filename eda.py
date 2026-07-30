

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/energy_dataset.csv")

print(df.shape)
print(df.columns.tolist())

df.head()

print(df.shape)
print(df.columns.tolist())


df['time'] = pd.to_datetime(df['time'], utc=True)
one_day = df[df['time'].dt.date == df['time'].dt.date.iloc[24]]


plt.plot(one_day['time'], one_day['generation solar'])
plt.title('Solar generation over one day')
plt.xlabel('Hour')
plt.ylabel('MW')
plt.show()


plt.plot(one_day['time'], one_day['generation solar'], label='Actual')
plt.plot(one_day['time'], one_day['forecast solar day ahead'], label='Grid operator forecast')
plt.title('Actual vs REE\'s own forecast')
plt.xlabel('Hour')
plt.ylabel('MW')
plt.legend()
plt.show()

df['error'] = df['generation solar'] - df['forecast solar day ahead']
df.groupby(df['time'].dt.hour)['error'].mean().plot(kind='bar')
plt.title('Average forecast error by hour of day')
plt.xlabel('Hour')
plt.ylabel('Actual minus forecast (MW)')
plt.show()

df['month'] = df['time'].dt.month

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

df['season'] = df['month'].apply(get_season)

seasonal_bias = df.groupby(['season', df['time'].dt.hour])['error'].mean().unstack(level=0)
seasonal_bias.plot(kind='line', figsize=(8,5))
plt.title('Average forecast error by hour, split by season')
plt.xlabel('Hour')
plt.ylabel('Actual minus forecast (MW)')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.show()

df['year'] = df['time'].dt.year

yearly_bias = df.groupby(['year', df['time'].dt.hour])['error'].mean().unstack(level=0)
yearly_bias.plot(kind='line', figsize=(8,5))
plt.title('Average forecast error by hour, split by year')
plt.xlabel('Hour')
plt.ylabel('Actual minus forecast (MW)')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.show()
