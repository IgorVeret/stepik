'''Какое качество у вас получилось?
Подберём параметры нашей модели
Переберите по сетке от 1 до 10 параметр числа соседей
Также вы попробуйте использоввать различные метрики: ['manhattan', 'euclidean']
Попробуйте использовать различные стратегии вычисления весов: [‘uniform’, ‘distance’]'''

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

all_data = pd.read_csv('forest_dataset.csv')
all_data.head()
labels = all_data[all_data.columns[-1]].values
feature_matrix = all_data[all_data.columns[:-1]].values
train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(
    feature_matrix, labels, test_size=0.2, random_state=42)
# создание модели

clf = KNeighborsClassifier()

# обучение модели

clf.fit(train_feature_matrix, train_labels)

# предсказание на тестовой выборке

y_pred = clf.predict(test_feature_matrix)

# получаем качество модели
print('Ответ на вопрос 1. Доля правильных ответов на обучении:', accuracy_score(test_labels, y_pred))
