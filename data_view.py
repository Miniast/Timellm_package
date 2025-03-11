import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 30)

df = pd.read_csv('./dataset/total.csv')
