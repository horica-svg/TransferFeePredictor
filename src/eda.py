import pandas as pd

df = pd.read_csv('data/raw/players.csv')

print(df.head())
print(df.info())
print(df['market_value_in_eur'].describe())