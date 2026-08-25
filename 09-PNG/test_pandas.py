# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'd:\CAAS\09-PNG\.venv\Lib\site-packages')

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_excel(r'd:\CAAS\09-PNG\植株数据.xlsx')
print('Column names:', df.columns.tolist())
print('Data shape:', df.shape)
print('First rows:')
print(df.head(10))
print('Data types:')
print(df.dtypes)
