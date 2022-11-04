'''Какая прогнозируемая вероятность pred_freq класса под номером 3 (до 2 знаков после запятой)?'''
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

all_data = pd.read_csv('forest_dataset.csv')
labels = all_data[all_data.columns[-1]].values
feature_matrix = all_data[all_data.columns[:-1]].values

# создание модели
clf = KNeighborsClassifier()

# Задать параметры
params = {'n_neighbors': range(1, 11), 'metric': ['manhattan', 'euclidean'], 'weights': ['uniform', 'distance']}
clf_grid = GridSearchCV(clf, params, cv=5, scoring='accuracy', n_jobs=-1)
clf_grid.fit(feature_matrix, labels)

print(clf_grid.best_params_)
