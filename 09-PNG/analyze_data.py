# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_excel(r'd:\CAAS\09-PNG\植株数据.xlsx')
print('Column names:', df.columns.tolist())
print('Data shape:', df.shape)
print('First rows:')
print(df.head(10))
print('Data types:')
print(df.dtypes)
