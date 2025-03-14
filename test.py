import pandas as pd

df = pd.read_csv('./result/lstm_result.csv')
print(df.describe())