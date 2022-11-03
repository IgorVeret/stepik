'''4). Какую долю (вещественное число от 0 до 1, округлить до 4-го знака) от всех покупателей составляют ВМЕСТЕ
 мужчины от 26 до 35 лет и женщины старше 36 лет (то есть нужно учесть несколько возрастных категорий)?
(речь не об уникальных ID, а о количестве таких строк)'''

import pandas as pd
import numpy as np

data = pd.read_csv('https://drive.google.com/uc?id=1JY8l5nSu9O4GtMDpaOfH2Oowlpza3U-c')
x=round(len(data.query("Gender=='M' & Age=='26-35' | Gender=='F' & Age>'36'")) / len(data), 4)
print(x)