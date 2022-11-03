'''3). Сколько NaN'ов в столбце Product_Category_3?'''

import pandas as pd
import numpy as np

data = pd.read_csv('https://drive.google.com/uc?id=1JY8l5nSu9O4GtMDpaOfH2Oowlpza3U-c')
x=data.Product_Category_3.isna().sum()
print(x)