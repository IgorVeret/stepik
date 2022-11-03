'''1). Сколько строк с мужчинами из категории города A? (речь не об уникальных ID мужчин, а о количестве строк)'''
import pandas as pd
import numpy as np

data = pd.read_csv('https://drive.google.com/uc?id=1JY8l5nSu9O4GtMDpaOfH2Oowlpza3U-c')
x=len(data[(data['Gender'] == 'M') & (data['City_Category']=='A')])
print(x)