'''2). Сколько женщин от 46 до 50, потративших (столбец Purchase) больше 20000 (условных единиц, в данном случае)?
(речь не об уникальных ID, а о количестве строк)'''

import pandas as pd
import numpy as np

data = pd.read_csv('https://drive.google.com/uc?id=1JY8l5nSu9O4GtMDpaOfH2Oowlpza3U-c')
x=len(data[(data.Gender == 'F') & (data.Purchase > 20000) & (data.Age == '46-50')])
print(x)